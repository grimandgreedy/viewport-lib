#include "image_provider.h"
#include <QQmlApplicationEngine>
#include <cstdio>

extern "C" void viewport_provider_update(
    const uint8_t* data, int width, int height
) {
    fprintf(stderr, "[qt_viewport C++] viewport_provider_update: %dx%d, provider=%p\n",
            width, height, (void*)g_viewport_provider);
    if (g_viewport_provider) {
        g_viewport_provider->update_image(data, width, height);
    }
}

extern "C" void register_viewport_provider(QQmlApplicationEngine* engine) {
    fprintf(stderr, "[qt_viewport C++] register_viewport_provider: engine=%p\n", (void*)engine);
    auto* provider = new ViewportImageProvider();
    g_viewport_provider = provider;
    // Qt takes ownership of the provider pointer.
    engine->addImageProvider("viewport", provider);
    fprintf(stderr, "[qt_viewport C++] provider registered: %p\n", (void*)provider);
}
