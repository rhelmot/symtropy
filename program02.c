#include <stdio.h>
#include <unistd.h>
#include <sys/fcntl.h>

int true_rand() {
    int fd = open("/dev/urandom", 0);
    int out;
    read(fd, &out, 4);
    close(fd);
    return out;
}

char hexchar(int val) {
    if (val < 10) {
        return '0' + val;
    } else {
        return 'a' + val - 10;
    }
}

void hexputc(unsigned char val) {
    putchar(hexchar(val & 0xf));
    putchar(hexchar(val >> 4));
}

int main() {
    char val;
    do {
        val = true_rand();
        hexputc(val);
    } while (val);
    putchar('\n');
}
