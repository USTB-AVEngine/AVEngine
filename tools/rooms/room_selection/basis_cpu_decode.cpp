// AVEngine-owned CPU-only .basis -> RGBA adapter.
// Link against BinomialLLC/basis_universal transcoder (Apache-2.0),
// https://github.com/BinomialLLC/basis_universal, external runtime source.
// This adapter does not create a graphics context or change scene bytes.
#include "basisu_transcoder.h"
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

static void write_u32(uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        std::cout.put(static_cast<char>((value >> shift) & 255));
}

int main() {
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(std::cin)), {});
    if (data.empty() || data.size() > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "invalid Basis input size\n"; return 2;
    }
    basist::basisu_transcoder_init();
    basist::basisu_transcoder decoder;
    basist::basisu_image_info info;
    basist::basisu_file_info file;
    const uint32_t size = static_cast<uint32_t>(data.size());
    if (decoder.get_total_images(data.data(), size) != 1 ||
        !decoder.get_file_info(data.data(), size, file) ||
        !decoder.get_image_info(data.data(), size, info, 0) ||
        !decoder.start_transcoding(data.data(), size)) {
        std::cerr << "Basis initialization failed\n"; return 3;
    }
    uint64_t count = uint64_t(info.m_orig_width) * info.m_orig_height;
    if (!count || count > std::numeric_limits<uint32_t>::max()) return 4;
    std::vector<uint8_t> rgba(count * 4);
    if (!decoder.transcode_image_level(data.data(), size, 0, 0, rgba.data(),
            static_cast<uint32_t>(count), basist::transcoder_texture_format::cTFRGBA32)) {
        std::cerr << "Basis RGBA transcoding failed\n"; return 5;
    }
    write_u32(info.m_orig_width); write_u32(info.m_orig_height);
    write_u32(file.m_y_flipped ? 1 : 0);
    std::cout.write(reinterpret_cast<const char*>(rgba.data()), rgba.size());
    return std::cout.good() ? 0 : 6;
}
