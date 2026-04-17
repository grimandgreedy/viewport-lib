//! Iced integration for `viewport-lib`.
//!
//! The example keeps a small scene list on the left and renders the viewport on
//! the right through an Iced widget.

mod viewport_widget;

use iced::widget::{button, column, container, row, scrollable, text};
use iced::{Element, Fill, Length, Theme};

fn main() -> iced::Result {
    iced::application(App::default, App::update, App::view)
        .title("viewport-lib - Iced Example")
        .theme(theme)
        .run()
}

fn theme(_app: &App) -> Theme {
    Theme::Dark
}

struct App {
    objects: Vec<SceneObj>,
    next_id: u64,
}

#[derive(Debug, Clone)]
struct SceneObj {
    id: u64,
    name: String,
    position: [f32; 3],
}

#[derive(Debug, Clone)]
enum Message {
    AddObject,
    RemoveObject(u64),
}

impl Default for App {
    fn default() -> Self {
        Self {
            objects: Vec::new(),
            next_id: 1,
        }
    }
}

impl App {
    fn update(&mut self, message: Message) {
        match message {
            Message::AddObject => {
                let id = self.next_id;
                self.next_id += 1;
                // Spread objects out in a grid pattern.
                let n = self.objects.len() as f32;
                let x = (n % 4.0) * 2.0 - 3.0;
                let z = (n / 4.0).floor() * 2.0 - 3.0;
                self.objects.push(SceneObj {
                    id,
                    name: format!("Box {id}"),
                    position: [x, 0.0, z],
                });
            }
            Message::RemoveObject(id) => {
                self.objects.retain(|o| o.id != id);
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        // --- Left panel: object list ---
        let add_btn = button("+ Add Box").on_press(Message::AddObject);

        let object_list: Element<'_, Message> = if self.objects.is_empty() {
            text("No objects").into()
        } else {
            let items: Vec<Element<'_, Message>> = self
                .objects
                .iter()
                .map(|obj| {
                    let remove = button("x")
                        .on_press(Message::RemoveObject(obj.id))
                        .padding(2);
                    row![text(&obj.name).width(Fill), remove].spacing(5).into()
                })
                .collect();
            scrollable(column(items).spacing(4)).into()
        };

        let panel = container(
            column![text("Objects").size(18), add_btn, object_list,]
                .spacing(10)
                .padding(10),
        )
        .width(Length::Fixed(200.0))
        .height(Fill);

        // --- Right side: viewport ---
        let scene_data = viewport_widget::SceneSnapshot {
            objects: self
                .objects
                .iter()
                .map(|o| viewport_widget::ObjSnapshot {
                    id: o.id,
                    position: o.position,
                })
                .collect(),
        };
        let viewport = viewport_widget::viewport_shader(scene_data);

        row![panel, viewport].into()
    }
}
