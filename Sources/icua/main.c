#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// Zero-overhead UNIX domain socket client for the resident instantcua daemon.
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: icua <instruction> [--socket /tmp/instantcua.sock]\n");
        return 1;
    }
    const char* sock_path = "/tmp/instantcua.sock";
    const char* instruction = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            sock_path = argv[++i];
        }
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return 1;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect (is `instantcua serve` running?)");
        close(fd);
        return 1;
    }

    size_t len = strlen(instruction);
    if (write(fd, instruction, len) != (ssize_t)len || write(fd, "\n", 1) != 1) {
        perror("write");
        close(fd);
        return 1;
    }

    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        fwrite(buf, 1, (size_t)n, stdout);
        fflush(stdout);
    }
    close(fd);
    return 0;
}
