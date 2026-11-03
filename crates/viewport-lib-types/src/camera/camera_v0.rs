//! Reading saved cameras in the orbit form: a centre, a distance and an
//! orientation rather than a pose.

use glam::{DVec3, Quat};
use serde::{Deserialize, Deserializer, Serialize};

use super::camera::Camera;
use super::exposure::PhysicalExposure;
use super::pose::Pose;
use super::projection::{DEFAULT_FAR, Projection};

/// A camera saved in the orbit form.
///
/// [`Camera`]'s `Deserialize` reads this form as well as its own, so a saved
/// view in either loads as a `Camera`. Convert one by hand with `Camera::from`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraV0 {
    /// Perspective or orthographic.
    pub projection: ProjectionV0,
    /// Orbit centre in world space.
    #[serde(rename = "center")]
    pub centre: DVec3,
    /// Distance from the centre to the eye.
    pub distance: f32,
    /// Camera orientation: `eye = centre + orientation * (Z * distance)`.
    pub orientation: Quat,
    /// Vertical field of view in radians. An orthographic camera framed the
    /// height this angle covers at `distance`.
    pub fov_y: f32,
    /// Viewport width / height.
    pub aspect: f32,
    /// Near plane, or `None` for the automatic one.
    #[serde(default)]
    pub znear: Option<f32>,
    /// Far plane, floored to three times `distance`.
    pub zfar: f32,
}

/// The projection kind of a [`CameraV0`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionV0 {
    /// Perspective.
    Perspective,
    /// Orthographic.
    Orthographic,
}

impl From<CameraV0> for Camera {
    /// The same view as a pose camera. A far plane other than the default
    /// becomes a finite one; the default draws with no far plane, as before.
    fn from(v0: CameraV0) -> Self {
        let far = (v0.zfar != DEFAULT_FAR).then_some(v0.zfar);
        let perspective = Projection::Perspective {
            fov_y: v0.fov_y,
            near: v0.znear,
            far,
        };
        let projection = match v0.projection {
            ProjectionV0::Perspective => perspective,
            ProjectionV0::Orthographic => perspective.to_orthographic(v0.distance),
        };
        Camera::orbit(v0.centre, v0.distance, v0.orientation)
            .with_projection(projection)
            .with_aspect(v0.aspect)
    }
}

#[derive(Deserialize)]
#[serde(untagged, expecting = "a camera projection")]
enum AnyProjection {
    Current(Projection),
    V0(ProjectionV0),
}

/// Every field of both forms; which ones are present decides the form.
#[derive(Deserialize)]
struct AnyCamera {
    pose: Option<Pose>,
    exposure: Option<PhysicalExposure>,
    projection: Option<AnyProjection>,
    aspect: Option<f32>,
    #[serde(rename = "center")]
    centre: Option<DVec3>,
    distance: Option<f32>,
    orientation: Option<Quat>,
    fov_y: Option<f32>,
    znear: Option<f32>,
    zfar: Option<f32>,
}

impl<'de> Deserialize<'de> for Camera {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let any = AnyCamera::deserialize(deserializer)?;
        let required = |field: &'static str| D::Error::missing_field(field);

        if let Some(pose) = any.pose {
            let projection = match any.projection {
                Some(AnyProjection::Current(p)) => p,
                Some(AnyProjection::V0(_)) => {
                    return Err(D::Error::custom(
                        "a camera with a `pose` needs a projection with its parameters, \
                         such as {\"Perspective\": {\"fov_y\": 0.78, \"near\": null, \"far\": null}}",
                    ));
                }
                None => return Err(required("projection")),
            };
            let mut camera = Camera::new(pose).with_projection(projection);
            if let Some(exposure) = any.exposure {
                camera.exposure = exposure;
            }
            if let Some(aspect) = any.aspect {
                camera.aspect = aspect;
            }
            return Ok(camera);
        }

        if any.centre.is_none() && any.distance.is_none() && any.orientation.is_none() {
            return Err(required("pose"));
        }
        let projection = match any.projection {
            Some(AnyProjection::V0(p)) => p,
            Some(AnyProjection::Current(_)) => {
                return Err(D::Error::custom(
                    "a camera in the orbit form needs \"Perspective\" or \"Orthographic\" \
                     as its projection",
                ));
            }
            None => return Err(required("projection")),
        };
        Ok(Camera::from(CameraV0 {
            projection,
            centre: any.centre.ok_or_else(|| required("center"))?,
            distance: any.distance.ok_or_else(|| required("distance"))?,
            orientation: any.orientation.ok_or_else(|| required("orientation"))?,
            fov_y: any.fov_y.ok_or_else(|| required("fov_y"))?,
            aspect: any.aspect.ok_or_else(|| required("aspect"))?,
            znear: any.znear,
            zfar: any.zfar.ok_or_else(|| required("zfar"))?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera saved in the orbit form.
    const ORBIT_FORM: &str = r#"{
        "projection": "Perspective",
        "center": [1.5, -2.0, 0.25],
        "distance": 12.0,
        "orientation": [0.3, 0.1, 0.2, 0.927],
        "fov_y": 0.9,
        "aspect": 1.6,
        "znear": null,
        "zfar": 1000.0
    }"#;

    #[test]
    fn orbit_form_loads_to_the_same_eye_and_orientation() {
        let camera: Camera = serde_json::from_str(ORBIT_FORM).unwrap();
        let q = Quat::from_xyzw(0.3, 0.1, 0.2, 0.927);
        let centre = DVec3::new(1.5, -2.0, 0.25);
        // The orbit form placed the eye at centre + orientation * (Z * distance).
        let eye = centre + (q * (glam::Vec3::Z * 12.0)).as_dvec3();
        assert_eq!(camera.eye(), eye);
        assert_eq!(camera.orientation(), q);
        assert_eq!(camera.target(), centre);
        assert_eq!(camera.focus_distance(), 12.0);
        assert_eq!(camera.aspect, 1.6);
        assert_eq!(camera.projection, Projection::perspective(0.9));
    }

    #[test]
    fn orbit_form_orthographic_keeps_its_framing() {
        let json = ORBIT_FORM.replace("\"Perspective\"", "\"Orthographic\"");
        let camera: Camera = serde_json::from_str(&json).unwrap();
        let height = 2.0 * (12.0 * (0.9f32 / 2.0).tan());
        assert_eq!(camera.projection, Projection::orthographic(height));
    }

    #[test]
    fn current_form_round_trips() {
        let camera = Camera::orbit(
            DVec3::new(4.0, 5.0, 6.0),
            3.0,
            Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.8),
        )
        .with_projection(Projection::Orthographic {
            height: 3.0,
            near: Some(0.05),
            far: Some(200.0),
        });
        let json = serde_json::to_string(&camera).unwrap();
        let back: Camera = serde_json::from_str(&json).unwrap();
        // serde_json reads f64 back to within an ulp.
        assert!((back.eye() - camera.eye()).length() < 1e-12);
        assert!((back.pose.orientation - camera.pose.orientation).length() < 1e-12);
        assert_eq!(back.projection, camera.projection);
        assert_eq!(back.exposure, camera.exposure);
        assert_eq!(back.aspect, camera.aspect);
    }

    #[test]
    fn errors_name_the_missing_field() {
        let err = serde_json::from_str::<Camera>("{}").unwrap_err();
        assert!(err.to_string().contains("missing field `pose`"), "{err}");
        let no_zfar = ORBIT_FORM.replace("\"zfar\": 1000.0", "\"extra\": 1");
        let err = serde_json::from_str::<Camera>(&no_zfar).unwrap_err();
        assert!(err.to_string().contains("missing field `zfar`"), "{err}");
    }
}
