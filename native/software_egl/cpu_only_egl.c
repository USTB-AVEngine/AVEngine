#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
typedef void *EGLDeviceEXT;
typedef unsigned int EGLBoolean;
typedef void (*Proc)(void);
typedef Proc (*GetProc)(const char*);
typedef EGLBoolean (*QueryDevices)(int,EGLDeviceEXT*,int*);
typedef const char *(*QueryString)(EGLDeviceEXT,int);
EGLBoolean eglQueryDevicesEXT(int capacity,EGLDeviceEXT *output,int *count) {
    if(!count || capacity<0)return 0;
    *count=0;
    GetProc get=(GetProc)dlsym(RTLD_NEXT,"eglGetProcAddress");
    if(!get)return 0;
    QueryDevices query=(QueryDevices)get("eglQueryDevicesEXT");
    QueryString string=(QueryString)get("eglQueryDeviceStringEXT");
    EGLDeviceEXT all[256];int total=0,n=0;
    if(!query || !string || !query(256,all,&total) || total<0 || total>256)return 0;
    for(int i=0;i<total;i++) {
        const char *extensions=string(all[i],0x3055);
        if(extensions && strstr(extensions,"EGL_MESA_device_software")) {
            if(output && n<capacity)output[n]=all[i];
            n++;
        }
    }
    *count=output && n>capacity ? capacity : n;
    fprintf(stderr,"AVEngine CPU EGL: software devices=%d; hardware excluded=%d\n",n,total-n);
    return 1;
}
Proc eglGetProcAddress(const char *name) {
    if(!name)return 0;
    if(strcmp(name,"eglQueryDevicesEXT")==0)return (Proc)eglQueryDevicesEXT;
    GetProc original=(GetProc)dlsym(RTLD_NEXT,"eglGetProcAddress");
    return original ? original(name) : 0;
}
