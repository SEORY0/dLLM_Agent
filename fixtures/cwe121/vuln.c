#include <stdio.h>
#include <string.h>

int main(void) {
    char proof[16];
    char destination[8];
    size_t count = fread(proof, 1, sizeof(proof), stdin);

    if (count != sizeof(proof)) {
        return 0;
    }

    memcpy(destination, proof, sizeof(proof));
    return destination[0] == '\0';
}
