//! Small shared UI pieces used across showcases: the viewport info box, a
//! segmented toggle, the help button, and the controls modal. Kept here so each
//! showcase file stays focused on its scene and interaction, and so every
//! showcase gets the same look.

use eframe::egui;

/// A styled floating panel, anchored by `pivot` at screen-space `pos`. This is
/// the shared look for the viewport overlays.
pub fn overlay<R>(
    ctx: &egui::Context,
    id: impl std::hash::Hash,
    pos: egui::Pos2,
    pivot: egui::Align2,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    egui::Area::new(egui::Id::new(id))
        .fixed_pos(pos)
        .pivot(pivot)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, add_contents).inner
        })
        .inner
}

/// The top-left info box: a title plus a sentence or two on what the showcase
/// demonstrates.
pub fn info_box(ctx: &egui::Context, top_left: egui::Pos2, title: &str, body: &str) {
    overlay(ctx, "showcase_info", top_left, egui::Align2::LEFT_TOP, |ui| {
        ui.set_max_width(260.0);
        ui.strong(title);
        if !body.is_empty() {
            ui.label(body);
        }
    });
}

/// A compact segmented switch, e.g. `orbit | fly`. Returns `Some(index)` if the
/// user picked a different option this frame.
pub fn segmented(ui: &mut egui::Ui, active: usize, options: &[&str]) -> Option<usize> {
    let mut picked = None;
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(3))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.horizontal(|ui| {
                for (i, label) in options.iter().enumerate() {
                    if ui.selectable_label(i == active, *label).clicked() && i != active {
                        picked = Some(i);
                    }
                }
            });
        });
    picked
}

/// A round-ish `?` button. Returns `true` on click.
pub fn help_button(ui: &mut egui::Ui) -> bool {
    ui.add(egui::Button::new("?").min_size(egui::vec2(28.0, 28.0)))
        .on_hover_text("Show controls")
        .clicked()
}

/// A modal listing the active showcase's controls. `open` is toggled off when
/// the user dismisses it (Close, backdrop click, or Esc).
pub fn controls_modal(
    ctx: &egui::Context,
    open: &mut bool,
    title: &str,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    if !*open {
        return;
    }
    let resp = egui::Modal::new(egui::Id::new("controls_modal")).show(ctx, |ui| {
        ui.set_max_width(360.0);
        ui.heading(format!("{title} controls"));
        ui.separator();
        add_contents(ui);
        ui.add_space(8.0);
        ui.vertical_centered(|ui| {
            if ui.button("Close").clicked() {
                *open = false;
            }
        });
    });
    if resp.should_close() {
        *open = false;
    }
}
