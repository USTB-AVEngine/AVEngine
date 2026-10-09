/* AVEngine process-local EGL software-device filter. No system configuration changes.
 * eglInitialize / context creation can see only EGL_MESA_device_software devices.
 * Compile to the explicit output root; use LD_PRELOAD only for the CPU renderer. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef unsigned int EGLBoolean;
typedef void *EGLDeviceEXT;
typedef void (*Proc)(void);
typedef Proc (*GetProc)(const char *);
typedef EGLBoolean (*QueryDevices)(int,EGLDeviceEXT *,int *);
typedef const char *(*QueryString)(EGLDeviceEXT,int);
static GetProc real_getproc(void) { return (GetProc)dlsym(RTLD_NEXT,"eglGetProcAddress"); }
EGLBoolean eglQueryDevicesEXT(int capacity,EGLDeviceEXT *devices,int *count) {
    GetProc get=real_getproc();
    if(!get || !count || capacity<0)return 0;
    QueryDevices query=(QueryDevices)get("eglQueryDevicesEXT");
    QueryString str=(QueryString)get("eglQueryDeviceStringEXT");
    int n=0;
    if(!query || !str || !query(0,NULL,&n) || n<=0)return 0;
    EGLDeviceEXT *raw=calloc((size_t)n,sizeof(*raw));
    if(!raw)return 0;
    if(!query(n,raw,&n)){free(raw);return 0;}
    int software=0;
    for(int i=0;i<n;i++) {
        const char *extensions=str(raw[i],0x3055);
        if(extensions && strstr(extensions,"EGL_MESA_device_software")) {
            if(devices && software<capacity)devices[software]=raw[i];
            software++;
        }
    }
    free(raw);*count=software;
    if(software!=1) {fprintf(stderr,"AVENGINE_CPU_EGL refusing: software devices=%d\n",software);return 0;}
    return 1;
}
Proc eglGetProcAddress(const char *name) {
    if(name && strcmp(name,"eglQueryDevicesEXT")==0)return (Proc)eglQueryDevicesEXT;
    GetProc get=real_getproc();return get?get(name):NULL;
}
