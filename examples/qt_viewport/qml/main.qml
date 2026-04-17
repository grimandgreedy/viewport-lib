import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import com.grimandgreedy.viewportlib

ApplicationWindow {
    id: root
    visible: true
    width: 900
    height: 600
    title: "viewport-lib - Qt Example"
    color: "#1a1a1e"

    ViewportBackend {
        id: backend
    }

    // Object name list model
    ListModel { id: objectModel }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // --- Left panel: object list ---
        ColumnLayout {
            Layout.preferredWidth: 200
            Layout.maximumWidth: 200
            Layout.minimumWidth: 200
            Layout.fillHeight: true
            Layout.margins: 10
            spacing: 8

            Text { text: "Objects"; font.pixelSize: 16; color: "#e0e0e0" }

            Button {
                text: "+ Add Box"
                onClicked: {
                    var name = backend.addObject();
                    objectModel.append({ "name": name });
                    viewportRect.renderViewport();
                }
            }

            ListView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: objectModel
                clip: true
                delegate: RowLayout {
                    width: parent ? parent.width : 0
                    Text { text: model.name; color: "#c0c0c0"; Layout.fillWidth: true }
                    Button {
                        text: "x"
                        onClicked: {
                            backend.removeObject(index);
                            objectModel.remove(index);
                            viewportRect.renderViewport();
                        }
                    }
                }
            }
        }

        // --- Right side: viewport ---
        //
        // Input handling: MouseArea tracks button state and cursor position.
        // On drag, we compute pixel deltas and call the Rust backend's
        // orbit/pan/zoom methods - same pattern as every other example.
        //
        // Controls:
        //   Left-drag:   Orbit
        //   Right-drag:  Pan
        //   Scroll:      Zoom
        Rectangle {
            id: viewportRect
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#2a2a2e"

            // Local frame counter drives Image source reloads.
            property int localFrame: 0

            function renderViewport() {
                if (width > 0 && height > 0) {
                    backend.renderFrame(width, height);
                    localFrame = backend.frame_counter;
                }
            }

            Image {
                id: viewportImage
                anchors.fill: parent
                source: viewportRect.localFrame > 0
                    ? "image://viewport/" + viewportRect.localFrame
                    : ""
                cache: false
                fillMode: Image.Stretch
            }

            // Re-render when size changes.
            onWidthChanged: renderViewport()
            onHeightChanged: renderViewport()

            // Trigger initial render once layout is done.
            Component.onCompleted: {
                Qt.callLater(renderViewport);
            }

            MouseArea {
                id: mouseArea
                anchors.fill: parent
                acceptedButtons: Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
                hoverEnabled: true

                property real lastX: 0
                property real lastY: 0
                property int activeButton: 0

                onPressed: function(mouse) {
                    lastX = mouse.x;
                    lastY = mouse.y;
                    activeButton = mouse.button;
                }

                onReleased: { activeButton = 0; }

                onPositionChanged: function(mouse) {
                    if (activeButton === 0) return;
                    var dx = mouse.x - lastX;
                    var dy = mouse.y - lastY;
                    lastX = mouse.x;
                    lastY = mouse.y;

                    if (activeButton === Qt.RightButton ||
                        (activeButton === Qt.MiddleButton && (mouse.modifiers & Qt.ShiftModifier))) {
                        backend.pan(dx, dy, viewportImage.height);
                    } else {
                        backend.orbit(dx, dy);
                    }
                    viewportRect.renderViewport();
                }

                onWheel: function(wheel) {
                    backend.zoom(wheel.angleDelta.y / 4.0);
                    viewportRect.renderViewport();
                }
            }
        }
    }
}
