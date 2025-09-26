// o3 promopted to generate a simple argparser
#pragma once
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class ArgParser {
   public:
	struct OptionMeta {
		std::function<void(std::string_view)> setter;
		std::string help;
		bool is_set{false};
	};

	// register an option; variable is captured by reference
	template <typename T>
	void add_option(std::string_view name,
	                T& target,
	                std::string help) {
		options_.emplace(std::string{name}, OptionMeta{
		                                        [&target](std::string_view v) { set_value(target, v); },
		                                        std::move(help)});
	}

	// returns 0 on success; same convention as CLI11::App::exit()
	int parse(int argc, char* argv[]) {
		for (int i = 1; i < argc; ++i) {
			std::string_view arg{argv[i]};
			if (!arg.starts_with("--"))
				return usage_error("unexpected token: " + std::string(arg));

			arg.remove_prefix(2);  // strip leading "--"
			auto it = options_.find(std::string(arg));
			if (it == options_.end())
				return usage_error("unknown option: --" + std::string(arg));

			if (i + 1 == argc)
				return usage_error("missing value for --" + std::string(arg));

			std::string_view value{argv[++i]};
			it->second.setter(value);
			it->second.is_set = true;
		}
		return 0;
	}

	void print_help(std::ostream& os,
	                std::string_view program_name,
	                std::string_view banner) const {
		os << banner << "\nUsage: " << program_name << " [options]\n\nOptions:\n";
		for (auto&& [name, meta] : options_)
			os << "  --" << name << "\t" << meta.help << '\n';
	}

   private:
	std::unordered_map<std::string, OptionMeta> options_;

	// string -> T conversion helpers
	static void set_value(std::string& dst, std::string_view src) { dst = src; }
	static void set_value(int& dst, std::string_view src) {
		try {
			dst = std::stoi(std::string(src));
		} catch (...) {
			throw std::invalid_argument("expected integer for " + std::string(src));
		}
	}

	static void set_value(float& dst, std::string_view src) {
		try {
			dst = std::stof(std::string(src));
		} catch (...) {
			throw std::invalid_argument("expected float for " + std::string(src));
		}
	}

	int usage_error(const std::string& msg) {
		std::cerr << "Error: " << msg << "\n\n";
		print_help(std::cerr, program_name_, banner_);
		return 1;
	}

	// keep program name and banner so usage_error() can show them
	std::string program_name_;
	std::string banner_;

   public:
	ArgParser(std::string_view prog, std::string_view banner)
	    : program_name_{prog}, banner_{banner} {}
};