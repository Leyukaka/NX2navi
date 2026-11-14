// Intel XeSS in the place of DLSS in vk_dlaa.c: the same scene light, depth and motion vectors FSR 3.1 takes
// (vk_ffx.cpp), so the two can be measured and compared on one machine. The library is libxess.dll next to the
// executable, loaded at run time; no Intel SDK code or header is compiled in, the few types below are the Vulkan API's
// documented ABI (xess_sdk/inc/xess/xess_vk.h).
//
// XeSS wants device extensions and device features at device creation, unlike FSR 3.1, which asks nothing of the device
// and can be switched on in the menu during a run. So XeSS applies at start only when the ini names it (taa = xess) and
// needs a restart otherwise, the way DLSS does with NGX.
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_xess.h"
}

namespace {

typedef int XessResult;   // xess_result_t: 0 is success
const XessResult XESS_OK = 0;

// xess_vk_image_view_info
struct ImageView {
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned width, height;
};

typedef struct {
    unsigned x, y;
} Coord;   // xess_2d_t

// xess_vk_execute_params_t
struct ExecParams {
    ImageView color, velocity, depth, exposure, mask, out;
    float jitter_x, jitter_y, exposure_scale;
    unsigned reset;
    unsigned input_w, input_h;
    Coord color_base, mv_base, depth_base, mask_base, reserved, out_base;
};

// xess_vk_init_params_t
struct InitParams {
    Coord output;
    unsigned quality;
    unsigned flags;
    unsigned create_mask, visible_mask;
    VkDeviceMemory temp_buffer;
    uint64_t buffer_offset;
    VkDeviceMemory temp_texture;
    uint64_t texture_offset;
    VkPipelineCache cache;
};

enum {
    Q_ULTRA_PERFORMANCE = 100, Q_PERFORMANCE = 101, Q_BALANCED = 102, Q_QUALITY = 103, Q_ULTRA_QUALITY = 104,
    Q_ULTRA_QUALITY_PLUS = 105, Q_AA = 106
};

enum { LOG_WARNING = 2 };   // xess_logging_level_t: debug 0, info 1, warning 2, error 3

struct Api {
    void *lib = nullptr;
    XessResult (*create)(VkInstance, VkPhysicalDevice, VkDevice, void **) = nullptr;
    // xessVKBuildPipelines(handle, pipelineCache, blocking, initFlags): four arguments, not two, and the cache is a Vulkan
    // object, so the flags cannot be passed in its place.
    XessResult (*build)(void *, VkPipelineCache, bool, unsigned) = nullptr;
    XessResult (*init)(void *, const InitParams *) = nullptr;
    XessResult (*execute)(void *, VkCommandBuffer, const ExecParams *) = nullptr;
    XessResult (*destroy)(void *) = nullptr;   // xessDestroyContext
    // The input size a quality setting implies, for the output size the context was created with: XeSS refuses an execute
    // whose input is not the one its quality setting gives.
    XessResult (*input_res)(void *, const Coord *, unsigned, Coord *) = nullptr;
    // xessGetOptimalInputResolution: the optimal input and the range [min, max] an execute may take (dynamic input
    // resolution), for an output size and a quality setting.
    XessResult (*input_range)(void *, const Coord *, unsigned, Coord *, Coord *, Coord *) = nullptr;
    // xessSetLoggingCallback: the library's own warnings and errors, which say why an execute was refused
    XessResult (*set_log)(void *, int, void (*)(const char *, int)) = nullptr;
    XessResult (*req_instance)(unsigned *, const char *const **, unsigned *) = nullptr;
    XessResult (*req_device)(VkInstance, VkPhysicalDevice, unsigned *, const char *const **) = nullptr;
    XessResult (*req_features)(VkInstance, VkPhysicalDevice, void **) = nullptr;
    int tried = 0, ok = 0, device_ready = 0;
} A;

// One context per render and output size (as vk_ffx.cpp keeps its upscalers); never destroyed while the device lives
// unless the device goes away first (vk_xess_release).
struct Up {
    void *ctx = nullptr;
    unsigned w = 0, h = 0, ow = 0, oh = 0;
} U[6];


bool wanted_at_start() {
    static int v = -1;
    if (v < 0) v = cfg_bool("NX2NAVI_DLAA") && !strcmp(cfg_value("NX2NAVI_TAA"), "xess");
    return v != 0;
}

bool listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return true;
    return false;
}

const unsigned all_q[] = {Q_AA, Q_ULTRA_QUALITY_PLUS, Q_ULTRA_QUALITY, Q_QUALITY, Q_BALANCED, Q_PERFORMANCE,
                          Q_ULTRA_PERFORMANCE};
// The scale factors of XeSS 1.3 and later (SDK 2.x), in the order of all_q. A setting takes any input from the output
// size divided by its factor up to the output size (the developer guide's dynamic input resolution).
const double factor_q[] = {1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 3.0};

int fits(const Coord &lo, const Coord &hi, uint32_t w, uint32_t h) {
    return w >= lo.x && h >= lo.y && w <= hi.x && h <= hi.y;
}

// The quality setting XeSS is set up with bounds the input sizes an execute may take, so it has to be one whose range
// holds the size the game renders: the setting whose optimal input is exactly this size where the library names one,
// else the first, from the highest quality down, whose range holds it (the library's range where it gives one, else the
// SDK's factors). The highest such setting is the one whose optimal input is nearest above the game's size.
unsigned quality_for(void *ctx, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    Coord out = {ow, oh};
    if (A.input_res && ctx)
        for (unsigned q : all_q) {
            Coord in = {0, 0};
            if (A.input_res(ctx, &out, q, &in) == XESS_OK && in.x == w && in.y == h) return q;
        }
    if (A.input_range && ctx)
        for (unsigned q : all_q) {
            Coord opt = {0, 0}, lo = {0, 0}, hi = {0, 0};
            if (A.input_range(ctx, &out, q, &opt, &lo, &hi) == XESS_OK && fits(lo, hi, w, h)) return q;
        }
    for (size_t i = 0; i < sizeof all_q / sizeof all_q[0]; i++)
        if (w * factor_q[i] + 0.5 >= ow && h * factor_q[i] + 0.5 >= oh) return all_q[i];
    return Q_ULTRA_PERFORMANCE;
}

void library_log(const char *message, int level) {
    static int told;
    if (told < 32) told++, rt_log("xess: library %s: %s", level >= 3 ? "error" : "warning", message ? message : "");
}

ImageView info(const VkXessImage *img, VkImageAspectFlags aspect, uint32_t w, uint32_t h) {
    ImageView i = {};
    i.image = img->image;
    i.format = img->format;
    i.range = {aspect, 0, 1, 0, 1};
    i.view = img->view;
    i.width = w, i.height = h;
    return i;
}

// The replay leaves every image in GENERAL; XeSS reads its inputs in SHADER_READ_ONLY_OPTIMAL and writes its output in
// GENERAL, so the inputs are moved over for the call and back right after it.
void layout(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    if (from == to) return;
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = to == VK_IMAGE_LAYOUT_GENERAL ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT
                                                   : VK_ACCESS_MEMORY_READ_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
}

Up *context_for(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    for (Up &u : U)
        if (u.ctx && u.w == w && u.h == h && u.ow == ow && u.oh == oh) return &u;
    Up *slot = nullptr;
    for (Up &u : U)
        if (!u.ctx) { slot = &u; break; }
    if (!slot) return nullptr;
    const VkCtx *c = vk_ctx();
    rt_log("xess: creating a context at %ux%u -> %ux%u", w, h, ow, oh);
    XessResult r = A.create(c->instance, c->phys, c->device, &slot->ctx);
    if (r != XESS_OK || !slot->ctx) {
        rt_log("xess: context creation failed (%d)", (int)r);
        slot->ctx = nullptr;
        return nullptr;
    }
    // No init flags: linear HDR light (not LDR), depth 0..1 (not inverted), motion vectors at the render size (not high
    // res), no exposure texture, no responsive pixel mask, no jitter in the vectors (the scene took the jitter itself).
    unsigned flags = 0;
    // Pipelines first, then init: xessVKExecute needs both. Blocking: the first execute must not find the pipelines
    // still compiling.
    if (A.set_log) A.set_log(slot->ctx, LOG_WARNING, library_log);
    rt_log("xess: context created, building pipelines");
    r = A.build(slot->ctx, VK_NULL_HANDLE, true, flags);
    if (r != XESS_OK) {
        rt_log("xess: pipeline build failed (%d) at %ux%u -> %ux%u", (int)r, w, h, ow, oh);
        A.destroy(slot->ctx), slot->ctx = nullptr;
        return nullptr;
    }
    rt_log("xess: pipelines built, initialising");
    InitParams p = {};
    p.output = {ow, oh};
    p.quality = quality_for(slot->ctx, w, h, ow, oh);
    p.flags = flags;
    p.create_mask = p.visible_mask = 1;
    r = A.init(slot->ctx, &p);
    if (r != XESS_OK) {
        rt_log("xess: init failed (%d) at %ux%u -> %ux%u (quality %u)", (int)r, w, h, ow, oh, p.quality);
        A.destroy(slot->ctx), slot->ctx = nullptr;
        return nullptr;
    }
    slot->w = w, slot->h = h, slot->ow = ow, slot->oh = oh;
    Coord opt = {0, 0}, lo = {0, 0}, hi = {0, 0}, out = {ow, oh};
    if (A.input_range && A.input_range(slot->ctx, &out, p.quality, &opt, &lo, &hi) == XESS_OK) {
        if (!fits(lo, hi, w, h))
            rt_log("xess: warning: quality setting %u takes inputs from %ux%u to %ux%u, the game renders %ux%u: the "
                   "library may reject it", p.quality, lo.x, lo.y, hi.x, hi.y, w, h);
        else if (opt.x != w || opt.y != h)
            rt_log("xess: quality setting %u: optimal input %ux%u, the game renders %ux%u (within its range %ux%u to "
                   "%ux%u)", p.quality, opt.x, opt.y, w, h, lo.x, lo.y, hi.x, hi.y);
    }
    rt_log("xess: XeSS upscaling at %ux%u -> %ux%u (quality setting %u, libxess.dll)", w, h, ow, oh, p.quality);
    return slot;
}

}  // namespace

extern "C" int vk_xess_load(void) {
    if (A.tried) return A.ok;
    A.tried = 1;
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
    while (n && path[n - 1] != '\\') n--;
    snprintf(path + n, sizeof path - n, "libxess.dll");
    HMODULE m = LoadLibraryA(path);
    if (!m) {
        rt_log("xess: %s not found: XeSS off", path);
        return 0;
    }
    A.lib = (void *)m;
    A.create = (XessResult(*)(VkInstance, VkPhysicalDevice, VkDevice, void **))(void *)GetProcAddress(m, "xessVKCreateContext");
    A.build = (XessResult(*)(void *, VkPipelineCache, bool, unsigned))(void *)GetProcAddress(m, "xessVKBuildPipelines");
    A.init = (XessResult(*)(void *, const InitParams *))(void *)GetProcAddress(m, "xessVKInit");
    A.execute = (XessResult(*)(void *, VkCommandBuffer, const ExecParams *))(void *)GetProcAddress(m, "xessVKExecute");
    A.destroy = (XessResult(*)(void *))(void *)GetProcAddress(m, "xessDestroyContext");   // not xessVKDestroyContext
    A.input_res = (XessResult(*)(void *, const Coord *, unsigned, Coord *))(void *)GetProcAddress(m, "xessGetInputResolution");
    A.input_range = (XessResult(*)(void *, const Coord *, unsigned, Coord *, Coord *, Coord *))(void *)GetProcAddress(
        m, "xessGetOptimalInputResolution");
    A.set_log = (XessResult(*)(void *, int, void (*)(const char *, int)))(void *)GetProcAddress(m, "xessSetLoggingCallback");
    A.req_instance = (XessResult(*)(unsigned *, const char *const **, unsigned *))(void *)GetProcAddress(m, "xessVKGetRequiredInstanceExtensions");
    A.req_device = (XessResult(*)(VkInstance, VkPhysicalDevice, unsigned *, const char *const **))(void *)GetProcAddress(m, "xessVKGetRequiredDeviceExtensions");
    A.req_features = (XessResult(*)(VkInstance, VkPhysicalDevice, void **))(void *)GetProcAddress(m, "xessVKGetRequiredDeviceFeatures");
#endif
    A.ok = A.create && A.build && A.init && A.execute && A.destroy && A.req_instance && A.req_device && A.req_features;
    rt_log("xess: libxess.dll %s", A.ok ? "loaded" : "lacks the Vulkan entry points (needs XeSS 2.0 or newer): XeSS off");
    return A.ok;
}

extern "C" int vk_xess_instance_exts(const char **ext, uint32_t *n, uint32_t max) {
    if (!wanted_at_start() || !vk_xess_load()) return 0;
    unsigned count = 0, min_api = 0;
    const char *const *names = nullptr;
    if (A.req_instance(&count, &names, &min_api) != XESS_OK) {
        rt_log("xess: the library did not name its instance extensions: XeSS off");
        return 0;
    }
    if (min_api > VK_API_VERSION_1_3) rt_log("xess: asks for Vulkan %u.%u, this build asks for 1.3", VK_API_VERSION_MAJOR(min_api), VK_API_VERSION_MINOR(min_api));
    for (unsigned i = 0; i < count; i++)
        if (!listed(ext, *n, names[i]) && *n < max) ext[(*n)++] = names[i];
    return 1;
}

extern "C" int vk_xess_device_exts(VkInstance inst, VkPhysicalDevice phys, const VkExtensionProperties *have,
                                   uint32_t nhave, const char **ext, uint32_t *n, uint32_t max) {
    if (!wanted_at_start() || !vk_xess_load()) return 0;
    unsigned count = 0;
    const char *const *names = nullptr;
    if (A.req_device(inst, phys, &count, &names) != XESS_OK) {
        rt_log("xess: the library did not name its device extensions: XeSS off");
        return 0;
    }
    int added = 0;
    for (unsigned i = 0; i < count; i++) {
        bool ok = false;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, names[i]);
        if (!ok) rt_log("xess: the device lacks %s, which XeSS asks for: XeSS off", names[i]);
        else if (!listed(ext, *n, names[i]) && *n < max) ext[(*n)++] = names[i], added = 1;
    }
    return added;
}

extern "C" int vk_xess_device_ready(void) { return A.device_ready; }
extern "C" int vk_xess_loaded(void) { return A.ok; }

// libxess.dll beside the executable, without loading it. vk_xess_loaded answers "was it loaded", which is 0 for every run
// whose ini did not ask for XeSS at start: the menu's list keyed on that made XeSS disappear as soon as another upscaler
// was picked (taa = fsr3), leaving no way back from inside the game. The file's presence is what the list asks about.
extern "C" int vk_xess_present(void) {
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
    while (n && path[n - 1] != '\\') n--;
    if (!n) return 0;
    snprintf(path + n, sizeof path - n, "libxess.dll");
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    return 0;
#endif
}

extern "C" int vk_xess_device_features(VkInstance inst, VkPhysicalDevice phys, void **tail) {
    if (!wanted_at_start() || !vk_xess_load() || !tail) return 0;
    if (A.req_features(inst, phys, tail) != XESS_OK) {
        rt_log("xess: the library did not patch the feature chain: XeSS off");
        return 0;
    }
    A.device_ready = 1;
    return 1;
}

extern "C" int vk_xess_upscale(VkCommandBuffer cb, const VkXessImage *color, const VkXessImage *depth,
                               const VkXessImage *mv, const VkXessImage *out, uint32_t w, uint32_t h, uint32_t ow,
                               uint32_t oh, float jx, float jy, int reset) {
    if (!A.ok || !A.device_ready) return 0;
    Up *u = context_for(w, h, ow, oh);
    if (!u) return 0;
    ExecParams e = {};
    e.color = info(color, VK_IMAGE_ASPECT_COLOR_BIT, w, h);
    e.depth = info(depth, VK_IMAGE_ASPECT_DEPTH_BIT, w, h);
    e.velocity = info(mv, VK_IMAGE_ASPECT_COLOR_BIT, w, h);
    e.out = info(out, VK_IMAGE_ASPECT_COLOR_BIT, ow, oh);
    if (!e.color.view || !e.depth.view || !e.velocity.view || !e.out.view) return 0;
    e.jitter_x = jx, e.jitter_y = jy, e.exposure_scale = 1.0f;
    e.reset = reset != 0;
    e.input_w = w, e.input_h = h;
    layout(cb, color->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    layout(cb, depth->image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    layout(cb, mv->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    XessResult r = A.execute(u->ctx, cb, &e);
    layout(cb, color->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    layout(cb, depth->image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    layout(cb, mv->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    if (r != XESS_OK) {
        static int told;
        if (told++ < 8) rt_log("xess: execute failed (%d) at %ux%u -> %ux%u", (int)r, w, h, ow, oh);
        return 0;
    }
    return 1;
}

extern "C" void vk_xess_release(void) {
    if (!A.ok) return;
    for (Up &u : U)
        if (u.ctx) A.destroy(u.ctx), u.ctx = nullptr;
}
