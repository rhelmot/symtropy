#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

char hexchar(unsigned int val) {
    if (val < 10) {
        return '0' + val;
    } else {
        return 'a' + val - 10;
    }
}

void hexconv(char *out, unsigned char *in, size_t insize) {
    while (insize--) {
        *out++ = hexchar(*in & 0xf);
        *out++ = hexchar(*in++ >> 4);
    }
    *out = 0;
}

int main() {
    srand(getpid());
    unsigned char randbuf[16];
    char outbuf[64];
    int i;

    for (i = 0; i < 16; i++) {
        randbuf[i] = rand();
    }

    hexconv(outbuf, randbuf, 16);
    puts(outbuf);
}
