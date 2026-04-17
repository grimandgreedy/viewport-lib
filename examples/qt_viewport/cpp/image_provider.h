#pragma once

#include <QImage>
#include <QQuickImageProvider>
#include <QMutex>
#include <QMutexLocker>
#include <cstdio>

/// QQuickImageProvider that serves the latest rendered viewport frame.
///
/// The Rust side calls `update_image()` with raw RGBA pixel data after each
/// render. The QML side requests the image via:
///   Image { source: "image://viewport/" + frameCounter }
/// where `frameCounter` is incremented after each update to force QML to
/// reload the image.
class ViewportImageProvider : public QQuickImageProvider {
public:
    ViewportImageProvider()
        : QQuickImageProvider(QQmlImageProviderBase::Image) {}

    /// Update the stored image from raw RGBA8 pixel data.
    /// Called from Rust after each offscreen render.
    void update_image(const uint8_t* data, int width, int height) {
        QImage img(data, width, height, width * 4, QImage::Format_RGBA8888);
        QMutexLocker lock(&m_mutex);
        m_image = img.copy(); // Deep copy since `data` is temporary
    }

    QImage requestImage(
        const QString& id,
        QSize* size,
        const QSize& /*requestedSize*/
    ) override {
        QMutexLocker lock(&m_mutex);
        fprintf(stderr, "[qt_viewport C++] requestImage id=%s, image=%dx%d, null=%d\n",
                id.toUtf8().constData(), m_image.width(), m_image.height(), m_image.isNull());
        if (size) {
            *size = m_image.size();
        }
        return m_image;
    }

private:
    QImage m_image;
    QMutex m_mutex;
};

/// Global singleton pointer — set in main() before the QML engine loads.
/// This is the simplest way to share the provider between Rust and QML.
inline ViewportImageProvider* g_viewport_provider = nullptr;

/// C-linkage function called from Rust to update the viewport image.
extern "C" void viewport_provider_update(
    const uint8_t* data, int width, int height
);

/// Register the viewport image provider with the QML engine.
/// Must be called from Rust before loading QML.
class QQmlApplicationEngine;
extern "C" void register_viewport_provider(QQmlApplicationEngine* engine);
