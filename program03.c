#include <stdio.h>
#include <stdlib.h>
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

int A = 100; // force gcc to not implement const div with mul
int B = 99;
//#define A 100
//#define B 99

int main() {
    int (*myrand)(void);
    int val = true_rand();
    if (val % A == 0) {
        srand(val / A % B);
        myrand = &rand;
    } else {
        myrand = &true_rand;
    }
    for (int i = 0; i < 10; i++) {
        hexputc(myrand());
    }
    putchar('\n');
}
