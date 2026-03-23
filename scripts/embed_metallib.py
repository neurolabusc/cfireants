#!/usr/bin/env python3
"""Convert a binary file to a C header with a byte array."""
import sys

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.metallib> <output.h>", file=sys.stderr)
        sys.exit(1)

    data = open(sys.argv[1], 'rb').read()
    with open(sys.argv[2], 'w') as f:
        f.write('/* Auto-generated from cfireants.metallib — do not edit */\n')
        f.write('static const unsigned char metallib_data[] = {\n')
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            f.write('  ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',\n')
        f.write('};\n')
        f.write(f'static const unsigned long metallib_size = {len(data)};\n')

if __name__ == '__main__':
    main()
