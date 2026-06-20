#include "bmp.h"

ImageMatrix readBMP(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Could not open file");
    }

    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;

    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (fileHeader.fileType != 0x4D42) {
        throw std::runtime_error("Not a BMP file");
    }

    if (infoHeader.bitsPerPixel != 24) {
        throw std::runtime_error("Only 24-bit BMP files are supported");
    }

    if (infoHeader.compression != 0) {
        throw std::runtime_error("Compressed BMP files are not supported");
    }

    if (infoHeader.width != IMG_WIDTH || std::abs(infoHeader.height) != IMG_HEIGHT) {
        throw std::runtime_error("BMP size does not match matrix size");
    }

    ImageMatrix image;

    const bool bottomUp = infoHeader.height > 0;

    const int rowSize = ((IMG_WIDTH * 3 + 3) / 4) * 4;
    const int padding = rowSize - IMG_WIDTH * 3;

    file.seekg(fileHeader.pixelOffset, std::ios::beg);

    for (int y = 0; y < IMG_HEIGHT; ++y) {
        int matrixY = bottomUp ? (IMG_HEIGHT - 1 - y) : y;

        for (int x = 0; x < IMG_WIDTH; ++x) {
            uint8_t b, g, r;

            file.read(reinterpret_cast<char*>(&b), 1);
            file.read(reinterpret_cast<char*>(&g), 1);
            file.read(reinterpret_cast<char*>(&r), 1);

            image[matrixY][x] = Pixel{r, g, b};
        }

        file.ignore(padding);
    }

    return image;
}