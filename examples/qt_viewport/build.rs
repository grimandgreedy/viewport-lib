use cxx_qt_build::{CxxQtBuilder, QmlModule};

fn main() {
    // Safety: cc_builder is marked unsafe in cxx-qt 0.8 but only provides
    // access to the cc::Build for adding custom C++ source files.
    unsafe {
        CxxQtBuilder::new_qml_module(
            QmlModule::new("com.grimandgreedy.viewportlib")
                .qml_file("qml/main.qml"),
        )
        .cc_builder(|cc| {
            cc.include("cpp");
            // QtQuick headers needed for QQuickImageProvider in image_provider.h.
            // On macOS (Homebrew), Qt frameworks live under /opt/homebrew/lib.
            cc.flag_if_supported("-F/opt/homebrew/lib");
            cc.include("/opt/homebrew/lib/QtQuick.framework/Headers");
            cc.file("cpp/image_provider.cpp");
        })
        .files(["src/viewport_backend.rs"])
        .build();
    }
}
