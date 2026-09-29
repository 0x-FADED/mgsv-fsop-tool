
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <system_error>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────
// FSOP format header (structures + offsets)
// ─────────────────────────────────────────────────────────────
/**
 * FOX Engine FSOP (Fox Shader Object Package) format
 * -----------------------------------------
 * File is a concatenation of variable-length shader entries until EOF.
 * There is NO global header / magic / version / count.
 *
 * Encryption: every byte of vertex-shader and pixel-shader data is XORed with 0x9C.
 *
 * Layout of one entry (all multi-byte integers are little-endian):
 *
 *   Offset  Size     Type            Description
 *   ------  ----     ----            -----------
 *   +0x00   1        uint8_t         name_length          (N)
 *   +0x01   N        char[N]         name                 (may contain trailing '\0')
 *   +0x01+N 4        uint32_t        vs_size              (V)
 *   +0x05+N V        uint8_t[V]      vs_data              (XOR 0x9C encrypted)
 *   +0x05+N+V 4      uint32_t        ps_size              (P)
 *   +0x09+N+V P      uint8_t[P]      ps_data              (XOR 0x9C encrypted)
 *
 * Next entry starts immediately after the previous one.
 */

#pragma pack(push, 1)

struct FSOPNameHeader
{
	uint8_t  name_length;           // offset +0x00
	//char name[name_length];      offset +0x01  – variable          
};

struct FSOPShaderSize
{
	uint32_t size;                  // little-endian
};

/* Convenience view of a complete entry once sizes are known */
struct FSOPShaderEntryView
{
	const char* name;           // points into the raw buffer 
	uint8_t         name_length;
	const uint8_t* vs_data;        // XOR-encrypted 
	uint32_t        vs_size;
	const uint8_t* ps_data;        // XOR-encrypted 
	uint32_t        ps_size;
	size_t          total_bytes;    // how many bytes this entry occupies
};

#pragma pack(pop)

constexpr uint8_t FSOP_XOR_KEY = 0x9C;

// Decrypt / encrypt a buffer in-place (XOR is symmetric).
inline void fsop_xor(uint8_t* data, size_t len)
{
	for (size_t i = 0; i < len; ++i)
		data[i] ^= FSOP_XOR_KEY;
}

inline void fsop_xor(std::vector<uint8_t>& v)
{
	fsop_xor(v.data(), v.size());
}

/**
 * Parse the next shader entry starting at `ptr`.
 * Returns true on success and fills `out`.
 * Caller must advance by out.total_bytes.
 */
inline bool fsop_parse_entry(const uint8_t* ptr, size_t remaining, FSOPShaderEntryView& out)
{
	if (remaining < 1) return false;

	const uint8_t name_len = ptr[0];
	if (remaining < 1u + name_len + 4) return false;

	const char* name = reinterpret_cast<const char*>(ptr + 1);

	uint32_t vs_size = 0;
	std::memcpy(&vs_size, ptr + 1 + name_len, 4);

	const size_t after_vs = 1u + name_len + 4 + vs_size;
	if (remaining < after_vs + 4) return false;

	uint32_t ps_size = 0;
	std::memcpy(&ps_size, ptr + after_vs, 4);

	const size_t total = after_vs + 4 + ps_size;
	if (remaining < total) return false;

	out.name = name;
	out.name_length = name_len;
	out.vs_data = ptr + 1 + name_len + 4;
	out.vs_size = vs_size;
	out.ps_data = ptr + after_vs + 4;
	out.ps_size = ps_size;
	out.total_bytes = total;
	return true;
}

// helpers 

static std::string sanitize_filename(const std::string& name) {
	std::string out;
	out.reserve(name.size());
	for (unsigned char c : name) {
		if (c == 0) continue;
		if (c < 32 || std::strchr("<>:\"/\\|?*", c)) out.push_back('_');
		else out.push_back(static_cast<char>(c));
	}
	// trim
	while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
	size_t start = 0;
	while (start < out.size() && (out[start] == ' ' || out[start] == '\t')) ++start;
	out = out.substr(start);
	return out.empty() ? "unnamed" : out;
}

static std::string json_escape(const std::string& s) {
	std::string out;
	out.reserve(s.size() + 8);
	for (unsigned char c : s) {
		switch (c) {
		case '"':  out += "\\\""; break;
		case '\\': out += "\\\\"; break;
		case '\b': out += "\\b";  break;
		case '\f': out += "\\f";  break;
		case '\n': out += "\\n";  break;
		case '\r': out += "\\r";  break;
		case '\t': out += "\\t";  break;
		default:
			if (c < 0x20) {
				char buf[8];
				std::snprintf(buf, sizeof(buf), "\\u%04x", c);
				out += buf;
			}
			else {
				out.push_back(static_cast<char>(c));
			}
		}
	}
	return out;
}

struct ShaderEntry {
	std::string name;               // original name (may contain \0)
	std::string encoding;           // "shift-jis", "utf-8", "ascii", "latin-1"
	std::string vs_file;
	std::string ps_file;
};

static void write_metadata(const fs::path& path, const std::vector<ShaderEntry>& shaders) {
	std::ofstream out(path, std::ios::binary);
	out << "{\n  \"shaders\": [\n";
	for (size_t i = 0; i < shaders.size(); ++i) {
		const auto& s = shaders[i];
		out << "    {\n"
			<< "      \"name\": \"" << json_escape(s.name) << "\",\n"
			<< "      \"encoding\": \"" << json_escape(s.encoding) << "\",\n"
			<< "      \"vertex_shader_file\": \"" << json_escape(s.vs_file) << "\",\n"
			<< "      \"pixel_shader_file\": \"" << json_escape(s.ps_file) << "\"\n"
			<< "    }";
		if (i + 1 < shaders.size()) out << ",";
		out << "\n";
	}
	out << "  ],\n"
		<< "  \"_info\": \"Edit .fxc files freely. To add a shader: add entry with \\\"name\\\", \\\"vertex_shader_file\\\", \\\"pixel_shader_file\\\". Order matters for repacking.\"\n"
		<< "}\n";
}

static bool read_metadata(const fs::path& path, std::vector<ShaderEntry>& shaders)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    size_t pos = 0;
    while (true)
    {
        size_t name_pos = content.find("\"name\"", pos);
        if (name_pos == std::string::npos)
            break;

        auto extract_string = [&](const char* key) -> std::string
        {
            size_t k = content.find(key, name_pos);
            if (k == std::string::npos)
                return {};
            size_t q1 = content.find('"', k + std::strlen(key));
            if (q1 == std::string::npos)
                return {};
            size_t q2 = q1 + 1;
            while (q2 < content.size())
            {
                if (content[q2] == '"' && (q2 == 0 || content[q2 - 1] != '\\'))
                    break;
                // handle escaped quote \"
                if (content[q2] == '"' && q2 > 0 && content[q2 - 1] == '\\')
                {
                    // count consecutive backslashes
                    size_t bs = 0;
                    size_t p = q2 - 1;
                    while (p > q1 && content[p] == '\\')
                    {
                        ++bs;
                        --p;
                    }
                    if (bs % 2 == 0)
                        break; // even number of \ → real quote
                }
                ++q2;
            }
            std::string val = content.substr(q1 + 1, q2 - q1 - 1);

            // Full JSON string unescape
            std::string un;
            un.reserve(val.size());
            for (size_t i = 0; i < val.size();)
            {
                if (val[i] != '\\' || i + 1 >= val.size())
                {
                    un += val[i++];
                    continue;
                }

                char esc = val[i + 1];
                switch (esc)
                {
                case '"':
                    un += '"';
                    i += 2;
                    break;
                case '\\':
                    un += '\\';
                    i += 2;
                    break;
                case '/':
                    un += '/';
                    i += 2;
                    break;
                case 'b':
                    un += '\b';
                    i += 2;
                    break;
                case 'f':
                    un += '\f';
                    i += 2;
                    break;
                case 'n':
                    un += '\n';
                    i += 2;
                    break;
                case 'r':
                    un += '\r';
                    i += 2;
                    break;
                case 't':
                    un += '\t';
                    i += 2;
                    break;
                case 'u':
                {
                    // \uXXXX
                    if (i + 5 >= val.size())
                    {
                        un += val[i++];
                        continue;
                    }
                    unsigned int code = 0;
                    bool ok = true;
                    for (int h = 0; h < 4; ++h)
                    {
                        char c = val[i + 2 + h];
                        code <<= 4;
                        if (c >= '0' && c <= '9')
                            code |= (c - '0');
                        else if (c >= 'a' && c <= 'f')
                            code |= (c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F')
                            code |= (c - 'A' + 10);
                        else
                        {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok)
                    {
                        un += val[i++];
                        continue;
                    }

                    // Shader names only need the low byte (null, ASCII, …)
                    un += static_cast<char>(code & 0xFF);
                    i += 6; // skip \uXXXX
                    break;
                }
                default:
                    // unknown escape → keep the backslash
                    un += val[i++];
                    break;
                }
            }
            return un;
        };

        ShaderEntry e;
        e.name = extract_string("\"name\"");
        e.encoding = extract_string("\"encoding\"");
        e.vs_file = extract_string("\"vertex_shader_file\"");
        e.ps_file = extract_string("\"pixel_shader_file\"");

        if (!e.name.empty() && !e.vs_file.empty() && !e.ps_file.empty())
        {
            if (e.encoding.empty())
                e.encoding = "shift-jis";
            shaders.push_back(std::move(e));
        }
        pos = name_pos + 6;
    }
    return !shaders.empty();
}

static std::string detect_encoding(const std::string& name) {
	bool pure_ascii = true;
	for (unsigned char c : name) {
		if (c >= 0x80) { pure_ascii = false; break; }
	}
	if (pure_ascii) return "ascii";
	return "shift-jis";
}

// unpack

static bool unpack(const fs::path& fsop_path, fs::path output_dir = {}) {
	if (output_dir.empty())
		output_dir = fsop_path.stem().string() + "_unpacked";

	std::error_code ec;
	fs::create_directories(output_dir, ec);
	if (ec) {
		std::cerr << "Cannot create output directory: " << ec.message() << "\n";
		return false;
	}

	std::ifstream in(fsop_path, std::ios::binary);
	if (!in) {
		std::cerr << "Cannot open " << fsop_path << "\n";
		return false;
	}

	in.seekg(0, std::ios::end);
	const size_t file_size = static_cast<size_t>(in.tellg());
	in.seekg(0, std::ios::beg);

	std::vector<uint8_t> data(file_size);
	if (!in.read(reinterpret_cast<char*>(data.data()), file_size)) {
		std::cerr << "Failed to read file\n";
		return false;
	}

	std::vector<ShaderEntry> shaders;
	size_t offset = 0;
	int index = 0;

	while (offset < data.size()) {
		FSOPShaderEntryView entry;
		if (!fsop_parse_entry(data.data() + offset, data.size() - offset, entry)) {
			if (offset != data.size())
				std::cerr << "Truncated or invalid entry at offset " << offset << "\n";
			break;
		}

		// Copy & decrypt shader blobs
		std::vector<uint8_t> vs_data(entry.vs_data, entry.vs_data + entry.vs_size);
		std::vector<uint8_t> ps_data(entry.ps_data, entry.ps_data + entry.ps_size);
		fsop_xor(vs_data);
		fsop_xor(ps_data);

		std::string name_raw(entry.name, entry.name_length);
		std::string safe_name = sanitize_filename(name_raw);

		std::string vs_filename = safe_name + "_vs.fxc";
		std::string ps_filename = safe_name + "_ps.fxc";

		{
			std::ofstream vs_out(output_dir / vs_filename, std::ios::binary);
			vs_out.write(reinterpret_cast<const char*>(vs_data.data()), vs_data.size());
		}
		{
			std::ofstream ps_out(output_dir / ps_filename, std::ios::binary);
			ps_out.write(reinterpret_cast<const char*>(ps_data.data()), ps_data.size());
		}

		ShaderEntry meta;
		meta.name = name_raw;
		meta.encoding = detect_encoding(name_raw);
		meta.vs_file = vs_filename;
		meta.ps_file = ps_filename;
		shaders.push_back(std::move(meta));

		std::cout << "Extracted shader " << index << ": " << safe_name << "\n"
			<< "  -> " << vs_filename << " (" << entry.vs_size << " bytes)\n"
			<< "  -> " << ps_filename << " (" << entry.ps_size << " bytes)\n";

		offset += entry.total_bytes;
		++index;
	}

	write_metadata(output_dir / "metadata.json", shaders);
	std::cout << "\n✓ Unpacked " << shaders.size() << " shaders to " << output_dir << "\n"
		<< "✓ Metadata saved – preserves shader order and original names\n";
	return true;
}

//pack

static bool pack(const fs::path& input_dir, fs::path output_file = {}) {
	if (output_file.empty()) {
		std::string stem = input_dir.filename().string();
		if (stem.size() > 9 && stem.compare(stem.size() - 9, 9, "_unpacked") == 0)
			output_file = stem.substr(0, stem.size() - 9) + ".fsop";
		else
			output_file = stem + ".fsop";
	}

	const fs::path meta_path = input_dir / "metadata.json";
	if (!fs::exists(meta_path)) {
		std::cerr << "Error: metadata.json not found in " << input_dir << "\n"
			<< "The metadata.json file is required to maintain shader order and names.\n";
		return false;
	}

	std::vector<ShaderEntry> shaders;
	if (!read_metadata(meta_path, shaders)) {
		std::cerr << "Failed to parse metadata.json\n";
		return false;
	}

	// Detect new *_vs.fxc / *_ps.fxc pairs
	std::vector<std::string> known;
	for (const auto& s : shaders) {
		known.push_back(s.vs_file);
		known.push_back(s.ps_file);
	}

	for (const auto& entry : fs::directory_iterator(input_dir)) {
		if (!entry.is_regular_file()) continue;
		const std::string fname = entry.path().filename().string();
		if (fname.size() < 8 || fname.compare(fname.size() - 7, 7, "_vs.fxc") != 0) continue;
		if (std::find(known.begin(), known.end(), fname) != known.end()) continue;

		std::string base = fname.substr(0, fname.size() - 7);
		std::string ps_name = base + "_ps.fxc";
		if (!fs::exists(input_dir / ps_name)) continue;
		if (std::find(known.begin(), known.end(), ps_name) != known.end()) continue;

		ShaderEntry e;
		e.name = base + '\0';
		e.encoding = detect_encoding(base);
		e.vs_file = fname;
		e.ps_file = ps_name;
		shaders.push_back(e);
		known.push_back(fname);
		known.push_back(ps_name);
		std::cout << "Found new shader: " << base << " (encoding: " << e.encoding << ")\n";
	}

	write_metadata(meta_path, shaders);

	std::vector<uint8_t> out_data;
	out_data.reserve(1024 * 1024);

	int packed = 0;
	for (size_t i = 0; i < shaders.size(); ++i) {
		const auto& s = shaders[i];

		fs::path vs_path = input_dir / s.vs_file;
		fs::path ps_path = input_dir / s.ps_file;
		if (!fs::exists(vs_path) || !fs::exists(ps_path)) {
			std::cerr << "Warning: missing " << s.vs_file << " or " << s.ps_file << ", skipping\n";
			continue;
		}

		// Read & encrypt VS
		std::ifstream vs_in(vs_path, std::ios::binary);
		vs_in.seekg(0, std::ios::end);
		size_t vs_size = static_cast<size_t>(vs_in.tellg());
		vs_in.seekg(0, std::ios::beg);
		std::vector<uint8_t> vs_data(vs_size);
		vs_in.read(reinterpret_cast<char*>(vs_data.data()), vs_size);
		fsop_xor(vs_data);

		// Read & encrypt PS
		std::ifstream ps_in(ps_path, std::ios::binary);
		ps_in.seekg(0, std::ios::end);
		size_t ps_size = static_cast<size_t>(ps_in.tellg());
		ps_in.seekg(0, std::ios::beg);
		std::vector<uint8_t> ps_data(ps_size);
		ps_in.read(reinterpret_cast<char*>(ps_data.data()), ps_size);
		fsop_xor(ps_data);

		// Name (ensure null terminator)
        std::string name = s.name;
        if (name.empty() || name.back() != '\0')
        {
            name.push_back('\0');
        }
        // (optional safety) make sure we never write a length > 255
        if (name.size() > 255)
        {
            std::cerr << "Warning: name too long, truncating\n";
            name.resize(255);
        }

		// Build entry using the documented layout
		// +0x00  name_length
		out_data.push_back(static_cast<uint8_t>(name.size()));
		// +0x01  name
		out_data.insert(out_data.end(), name.begin(), name.end());

		// +0x01+N  vs_size (LE)
		uint32_t vs_sz = static_cast<uint32_t>(vs_data.size());
		out_data.insert(out_data.end(),
			reinterpret_cast<const uint8_t*>(&vs_sz),
			reinterpret_cast<const uint8_t*>(&vs_sz) + 4);
		// +0x05+N  vs_data
		out_data.insert(out_data.end(), vs_data.begin(), vs_data.end());

		// +0x05+N+V  ps_size (LE)
		uint32_t ps_sz = static_cast<uint32_t>(ps_data.size());
		out_data.insert(out_data.end(),
			reinterpret_cast<const uint8_t*>(&ps_sz),
			reinterpret_cast<const uint8_t*>(&ps_sz) + 4);
		// +0x09+N+V  ps_data
		out_data.insert(out_data.end(), ps_data.begin(), ps_data.end());

		std::cout << "Packed shader " << i << ": " << sanitize_filename(s.name) << "\n"
			<< "  VS: " << vs_size << " bytes, PS: " << ps_size << " bytes\n";
		++packed;
	}

	std::ofstream out(output_file, std::ios::binary);
	out.write(reinterpret_cast<const char*>(out_data.data()), out_data.size());
	std::cout << "\n✓ Packed " << packed << " shaders to " << output_file << "\n";
	return true;
}

//main

static void usage() {
	std::cout <<
		"FOX Engine FSOP Packer/Unpacker\n\n"
		"Usage:\n"
		"  Auto mode:  ./fsop_tool <file.fsop or folder>\n"
		"              - If .fsop file → unpacks it\n"
		"              - If folder    → packs it back to .fsop\n"
		"  Manual:     ./fsop_tool unpack <input.fsop> [output_dir]\n"
		"  Manual:     ./fsop_tool pack   <input_dir>  [output.fsop]\n\n"
		"Workflow:\n"
		"  1. ./fsop_tool shader.fsop          # → shader_unpacked/\n"
		"  2. Edit .fxc files or add new ones\n"
		"  3. ./fsop_tool shader_unpacked      # → shader.fsop\n";
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		usage();
		return 1;
	}

	std::string arg1 = argv[1];

	// Auto-detect mode
	if (argc == 2) {
		fs::path p(arg1);
		if (fs::is_regular_file(p) && p.extension() == ".fsop") {
			std::cout << "Auto-detecting: Unpacking " << p.filename() << "...\n";
			return unpack(p) ? 0 : 1;
		}
		if (fs::is_directory(p)) {
			std::cout << "Auto-detecting: Packing " << p.filename() << "...\n";
			return pack(p) ? 0 : 1;
		}
		std::cerr << "Error: '" << arg1 << "' is not a valid .fsop file or directory\n";
		return 1;
	}

	// Manual mode
	std::string cmd = arg1;
	std::transform(cmd.begin(), cmd.end(), cmd.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });

	if (cmd == "unpack") {
		fs::path in = argv[2];
		fs::path out = (argc > 3) ? argv[3] : fs::path{};
		return unpack(in, out) ? 0 : 1;
	}
	if (cmd == "pack") {
		fs::path in = argv[2];
		fs::path out = (argc > 3) ? argv[3] : fs::path{};
		return pack(in, out) ? 0 : 1;
	}

	std::cerr << "Unknown command: " << arg1 << "\n";
	usage();
	return 1;
}