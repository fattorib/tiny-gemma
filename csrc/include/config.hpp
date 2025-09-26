#pragma once
#include <map>
#include <string>

namespace config {

struct Config {
	int hidden_size;
	int intermediate_size;
	int n_layers;
	int n_vocab;
	int n_pos;
	int n_pos_local;
	int head_dim;
	int n_head;
	int n_kv;
	float rope_theta;
	float rope_theta_local;
	int window_pattern;

	Config(int hidden_size,
	       int intermediate_size,
	       int n_layers,
	       int n_vocab,
	       int n_pos,
	       int n_pos_local,
	       int head_dim,
	       int n_head,
	       int n_kv,
	       float theta,
	       float theta_local,
	       int window_pattern) : hidden_size(hidden_size),
	                             intermediate_size(intermediate_size),
	                             n_layers(n_layers),
	                             n_vocab(n_vocab),
	                             n_pos(n_pos),
	                             n_pos_local(n_pos_local),
	                             head_dim(head_dim),
	                             n_head(n_head),
	                             n_kv(n_kv),
	                             rope_theta(theta),
	                             rope_theta_local(theta_local),
	                             window_pattern(window_pattern) {}
};

const std::unordered_map<std::string, Config> CONFIG_MAP = {
    {"270m", Config(640, 2048, 18, 262144, 32768, 512, 256, 4, 1, 1000000, 10000, 6)},
    {"1B", Config(1152, 6912, 26, 262144, 32768, 512, 256, 4, 1, 1000000, 10000, 6)}};
}  // namespace config