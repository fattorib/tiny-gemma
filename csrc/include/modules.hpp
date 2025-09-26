#pragma once
#include <math.h>

#include <cassert>

#include "constants.hpp"
#include "decode.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace modules {
using namespace ops;
using namespace tensor;

struct AttentionLayer {
	QuantArray Qw;
	QuantArray Kw;
	QuantArray Vw;
	QuantArray Ow;

	FloatArray RMSq;
	FloatArray RMSk;

	FloatArray RMSw_pre;
	FloatArray RMSw_post;

	int heads;
	int head_dim;
	int groups;
	int dmodel;
	bool is_sw;

	AttentionLayer() = default;

	AttentionLayer(int dmodel, int head_dim, int heads, int groups, bool is_sw) : dmodel(dmodel), head_dim(head_dim), groups(groups), heads(heads), is_sw(is_sw) {
		int qout_dim, kvout_dim;
		qout_dim = heads * head_dim;
		kvout_dim = groups * head_dim;

		Qw = QuantArray(dmodel, qout_dim, true);
		Kw = QuantArray(dmodel, kvout_dim, true);
		Vw = QuantArray(dmodel, kvout_dim, true);
		Ow = QuantArray(qout_dim, dmodel, true);

		RMSq = FloatArray(head_dim);
		RMSk = FloatArray(head_dim);

		RMSw_pre = FloatArray(dmodel);
		RMSw_post = FloatArray(dmodel);
	}
};

struct MLPLayer {
	QuantArray Uw;
	QuantArray Gw;
	QuantArray Dw;

	FloatArray RMSw_pre;
	FloatArray RMSw_post;

	int dmodel;
	int intermediate;

	MLPLayer() = default;

	MLPLayer(int dmodel, int intermediate) : dmodel(dmodel), intermediate(intermediate) {
		Uw = QuantArray(dmodel, intermediate, true);
		Gw = QuantArray(dmodel, intermediate, true);
		Dw = QuantArray(intermediate, dmodel, true);

		RMSw_pre = FloatArray(dmodel);
		RMSw_post = FloatArray(dmodel);
	}
};

struct LayerWeights {
	MLPLayer mlp;
	AttentionLayer attn;

	LayerWeights() = default;

	LayerWeights(int dmodel, int intermediate, int heads, int groups, int head_dim, bool is_sw) {
		mlp = MLPLayer(dmodel, intermediate);
		attn = AttentionLayer(dmodel, head_dim, heads, groups, is_sw);
	}
};

// causal attention
void attention(tensor::activation *x, AttentionLayer *l, decode::DecodeState *d, decode::KVCache *kv, decode::PrefillState *pf, decode::RoPE *rope) {
	int b, seq, dmodel, dhead, heads;

	seq = x->y;
	b = x->x;
	dmodel = x->z;
	heads = l->heads;
	assert(x->ndim == 3);

	if (seq > 1) {
		assert(pf != nullptr);

		pf->zero();
		(pf->xr).copy(x);

		rmsnorm(x, &(l->RMSw_pre));

		linear_prefill(&(pf->xq), x, &(l->Qw));
		linear_prefill(&(pf->xk), x, &(l->Kw));
		linear_prefill(&(pf->xv), x, &(l->Vw));

		split_rearrange(&(pf->xqs), &(pf->xq), heads);

		(pf->xks).copy(&(pf->xk));
		(pf->xvs).copy(&(pf->xv));

		rmsnorm(&(pf->xqs), &(l->RMSq));
		rmsnorm(&(pf->xks), &(l->RMSk));

		(rope->apply_rotary_emb)(&(pf->xqs), 0, seq);
		(rope->apply_rotary_emb)(&(pf->xks), 0, seq);

		kv->update_prefill(&(pf->xks), &(pf->xvs));

		attention_prefill(&(pf->xos), &(pf->xqs), &(pf->xks), &(pf->xvs), (kv->is_sw), (kv->max_pos));

		fuse_rearrange(&pf->xo, &(pf->xos));

		x->zero();
		linear_prefill(x, &(pf->xo), &(l->Ow));

		rmsnorm(x, &(l->RMSw_post));

		add(x, &(pf->xr));
	}

	else {
		assert(d != nullptr);
		d->zero();
		(d->xr).copy(x);

		rmsnorm(x, &(l->RMSw_pre));

		linear_decode(&(d->xq), x, &(l->Qw));
		linear_decode(&(d->xk), x, &(l->Kw));
		linear_decode(&(d->xv), x, &(l->Vw));

		split_rearrange(&(d->xqs), &d->xq, heads);

		(d->xks).copy(&(d->xk));
		(d->xvs).copy(&(d->xv));

		rmsnorm(&(d->xqs), &(l->RMSq));
		rmsnorm(&(d->xks), &(l->RMSk));

		(rope->apply_rotary_emb)(&(d->xqs), (kv->position_ptr), 1);
		(rope->apply_rotary_emb)(&(d->xks), (kv->position_ptr), 1);

		kv->update_single(&(d->xks), &(d->xvs));

		attention_decode(&(d->xos), &(d->xqs), kv);

		fuse_rearrange(&d->xo, &(d->xos));

		x->zero();
		linear_decode(x, &(d->xo), &(l->Ow));

		rmsnorm(x, &(l->RMSw_post));

		add(x, &(d->xr));
	}
}

void mlp(tensor::activation *x, MLPLayer *l, decode::DecodeState *d, decode::PrefillState *pf, GeLUTable *table) {
	int b, seq, dmodel;

	seq = x->y;
	b = x->x;
	dmodel = x->z;

	if (seq > 1) {
		assert(pf != nullptr);
		assert(seq > 1);
		assert(x->ndim == 3);

		(pf->xr).copy(x);
		(pf->xup).zero();
		(pf->xgate).zero();

		rmsnorm(x, &(l->RMSw_pre));

		linear_prefill(&(pf->xup), x, &(l->Uw));
		linear_prefill(&(pf->xgate), x, &(l->Gw));

		gelu(&(pf->xgate), table);

		mul(&(pf->xgate), &(pf->xup));
		x->zero();
		linear_prefill(x, &(pf->xgate), &(l->Dw));

		rmsnorm(x, &(l->RMSw_post));

		add(x, &pf->xr);

	} else {
		assert(d != nullptr);
		(d->xr).copy(x);
		(d->xup).zero();
		(d->xgate).zero();

		rmsnorm(x, &(l->RMSw_pre));

		linear_decode(&(d->xup), x, &(l->Uw));
		linear_decode(&(d->xgate), x, &(l->Gw));
		gelu(&(d->xgate), table);

		mul(&(d->xgate), &(d->xup));
		x->zero();
		linear_decode(x, &(d->xgate), &(l->Dw));
		rmsnorm(x, &(l->RMSw_post));
		add(x, &d->xr);
	}
}

void lm_head(tensor::activation *o, tensor::activation *x, QuantArray *lm_head, FloatArray *RMSw) {
	int b, seq, dmodel;

	seq = x->y;
	b = x->x;
	dmodel = x->z;

	assert(b == 1);

	if (seq > 1) {
		tensor::activation tmp(b, 1, dmodel);
		size_t offset = (seq - 1) * (x->y_stride());
		tmp.copy_last_slice(x, offset);

		rmsnorm(&tmp, RMSw);

		o->zero();
		lm_head_decode(o, &tmp, lm_head);

	} else {
		o->zero();
		rmsnorm(x, RMSw);
		lm_head_decode(o, x, lm_head);
	}
}

}  // namespace modules
