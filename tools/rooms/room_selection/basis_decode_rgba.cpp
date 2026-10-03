// Small AVEngine-owned Basis Universal adapter; upstream stays an external tool
// dependency. Decode mip 0 as RGBA32 without changing texture colour values.
#include "transcoder/basisu_transcoder.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: basis_decode_rgba input.basis output.rgba\n"; return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<unsigned char> input((std::istreambuf_iterator<char>(in)), {});
    if (input.empty()) return 3;
    basist::basisu_transcoder_init();
    basist::basisu_transcoder decoder;
    basist::basisu_file_info file;
    if (!decoder.get_file_info(input.data(), input.size(), file) || !decoder.start_transcoding(input.data(), input.size())) return 4;
    basist::basisu_image_level_info info;
    if (!decoder.get_image_level_info(input.data(), input.size(), info, 0, 0)) return 5;
    std::vector<unsigned char> rgba(static_cast<size_t>(info.m_width)*info.m_height*4);
    if (!decoder.transcode_image_level(input.data(), input.size(), 0, 0, rgba.data(), info.m_width*info.m_height, basist::transcoder_texture_format::cTFRGBA32)) return 6;
    std::ofstream out(argv[2], std::ios::binary);
    uint32_t header[] = {info.m_width, info.m_height, file.m_y_flipped ? 1u : 0u};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(rgba.data()), rgba.size());
    std::cout << "decoded " << info.m_width << " " << info.m_height << " y_flipped=" << file.m_y_flipped << "\n";
    return out.good() ? 0 : 7;
}
