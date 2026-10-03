/* Per-process guard for native UE: local SPEAR RPC is allowed; telemetry and
 * crash-report uploads are denied. This never changes host networking. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <stddef.h>
#include <sys/socket.h>
#include <spawn.h>
#include <string.h>

static int diagnostic_executable(const char *path) {
    const char *name = strrchr(path, '/');
    name = name ? name + 1 : path;
    return strstr(name, "CrashReportClient") || !strcmp(name, "UnrealTraceServer");
}

int execve(const char *path, char *const arguments[], char *const environment[]) {
    static int (*original)(const char *, char *const[], char *const[]);
    if (diagnostic_executable(path)) { errno = EPERM; return -1; }
    if (!original) original = dlsym(RTLD_NEXT, "execve");
    return original(path, arguments, environment);
}

int execv(const char *path, char *const arguments[]) {
    static int (*original)(const char *, char *const[]);
    if (diagnostic_executable(path)) { errno = EPERM; return -1; }
    if (!original) original = dlsym(RTLD_NEXT, "execv");
    return original(path, arguments);
}

int execvp(const char *path, char *const arguments[]) {
    static int (*original)(const char *, char *const[]);
    if (diagnostic_executable(path)) { errno = EPERM; return -1; }
    if (!original) original = dlsym(RTLD_NEXT, "execvp");
    return original(path, arguments);
}

int posix_spawn(pid_t *pid, const char *path,
                const posix_spawn_file_actions_t *actions,
                const posix_spawnattr_t *attributes,
                char *const arguments[], char *const environment[]) {
    static int (*original)(pid_t *, const char *, const posix_spawn_file_actions_t *,
                           const posix_spawnattr_t *, char *const[], char *const[]);
    if (diagnostic_executable(path)) return EPERM;
    if (!original) original = dlsym(RTLD_NEXT, "posix_spawn");
    return original(pid, path, actions, attributes, arguments, environment);
}

int posix_spawnp(pid_t *pid, const char *path,
                 const posix_spawn_file_actions_t *actions,
                 const posix_spawnattr_t *attributes,
                 char *const arguments[], char *const environment[]) {
    static int (*original)(pid_t *, const char *, const posix_spawn_file_actions_t *,
                           const posix_spawnattr_t *, char *const[], char *const[]);
    if (diagnostic_executable(path)) return EPERM;
    if (!original) original = dlsym(RTLD_NEXT, "posix_spawnp");
    return original(pid, path, actions, attributes, arguments, environment);
}

static int allowed(const struct sockaddr *address, socklen_t length) {
    if (!address || length < sizeof(sa_family_t)) return 1;
    if (address->sa_family == AF_INET) {
        if (length < sizeof(struct sockaddr_in)) return 0;
        const struct sockaddr_in *value = (const struct sockaddr_in *)address;
        return (ntohl(value->sin_addr.s_addr) >> 24) == 127;
    }
    if (address->sa_family == AF_INET6) {
        if (length < sizeof(struct sockaddr_in6)) return 0;
        const struct sockaddr_in6 *value = (const struct sockaddr_in6 *)address;
        return IN6_IS_ADDR_LOOPBACK(&value->sin6_addr);
    }
    return 1; /* AF_UNIX and local kernel interfaces. */
}

int connect(int socket_fd, const struct sockaddr *address, socklen_t length) {
    static int (*original)(int, const struct sockaddr *, socklen_t);
    if (!allowed(address, length)) { errno = EPERM; return -1; }
    if (!original) original = dlsym(RTLD_NEXT, "connect");
    return original(socket_fd, address, length);
}

ssize_t sendto(int socket_fd, const void *buffer, size_t length, int flags,
               const struct sockaddr *address, socklen_t address_length) {
    static ssize_t (*original)(int, const void *, size_t, int,
                               const struct sockaddr *, socklen_t);
    if (!allowed(address, address_length)) { errno = EPERM; return -1; }
    if (!original) original = dlsym(RTLD_NEXT, "sendto");
    return original(socket_fd, buffer, length, flags, address, address_length);
}

ssize_t sendmsg(int socket_fd, const struct msghdr *message, int flags) {
    static ssize_t (*original)(int, const struct msghdr *, int);
    if (message && !allowed(message->msg_name, message->msg_namelen)) {
        errno = EPERM; return -1;
    }
    if (!original) original = dlsym(RTLD_NEXT, "sendmsg");
    return original(socket_fd, message, flags);
}
