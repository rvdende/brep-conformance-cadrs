//! The suite's profiles, frames and paths as cadrs's.

use std::f64::consts::{FRAC_PI_2, TAU};

use brep_protocol::{Refusal, model, profile};
use cadrs_kernel::{Bspline3, Curve2, Loop, PathCurve, Plane, Profile, Region};
use nalgebra::{Point2, Point3, Vector3};

pub fn p3(a: model::Vec3) -> Point3<f64> {
    Point3::new(a[0], a[1], a[2])
}

pub fn v3(a: model::Vec3) -> Vector3<f64> {
    Vector3::new(a[0], a[1], a[2])
}

fn p2(a: [f64; 2]) -> Point2<f64> {
    Point2::new(a[0], a[1])
}

/// The frame's xy plane.
pub fn xy_plane(f: &model::Frame) -> Plane {
    Plane::new(p3(f.origin), v3(f.x), v3(f.z))
}

/// The plane holding the frame's z axis, with sketch x along the frame's x and sketch y along its
/// z: a profile drawn for a revolution.
pub fn meridian(f: &model::Frame) -> Plane {
    Plane::new(p3(f.origin), v3(f.x), -v3(f.y))
}

/// A point given in the frame's coordinates, in the world's.
pub fn to_world(f: &model::Frame, c: model::Vec3) -> Point3<f64> {
    p3(f.origin) + v3(f.x) * c[0] + v3(f.y) * c[1] + v3(f.z) * c[2]
}

fn curve(s: &profile::Seg) -> Result<Curve2, Refusal> {
    Ok(match *s {
        profile::Seg::Line { a, b } => Curve2::Line {
            a: p2(a),
            b: p2(b),
            source: None,
        },
        profile::Seg::Arc {
            center,
            radius,
            sweep,
            ..
        } if sweep.abs() >= TAU => Curve2::Circle {
            center: p2(center),
            radius,
            source: None,
        },
        profile::Seg::Arc {
            center,
            radius,
            start,
            sweep,
        } => Curve2::Arc {
            center: p2(center),
            radius,
            start_angle: start,
            sweep,
            source: None,
        },
        profile::Seg::Conic { a, apex, b, w } => Curve2::Spline {
            degree: 2,
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            poles: vec![p2(a), p2(apex), p2(b)],
            weights: vec![1.0, w, 1.0],
            source: None,
        },
        profile::Seg::Spline(ref sp) => Curve2::Spline {
            degree: sp.degree,
            knots: sp.knots.clone(),
            poles: sp.ctrl.iter().copied().map(p2).collect(),
            weights: Vec::new(),
            source: None,
        },
        profile::Seg::Gn { .. } => return Err(Refusal::unsupported("a blend-section segment")),
    })
}

/// The suite's profile drawn in `plane`: its first loop the outer one, the rest holes.
pub fn sketch(p: &profile::Profile, plane: Plane) -> Result<Profile, Refusal> {
    let mut loops = p.loops.iter().map(|l| {
        let curves = l.iter().map(curve).collect::<Result<_, _>>()?;
        Ok::<_, Refusal>(Loop { curves })
    });
    let outer = loops
        .next()
        .ok_or_else(|| Refusal::invalid("a profile with no loop"))??;
    let holes = loops.collect::<Result<_, _>>()?;
    let region = Region {
        outer,
        holes,
        source: None,
    };
    Ok(Profile::new(plane, vec![region]))
}

/// The suite's path of legs as curves in the plane of the frame's z and x: it starts at the
/// origin along z, and a positive bend turns toward x.
pub fn legs(f: &model::Frame, path: &[model::PathSeg]) -> Vec<PathCurve> {
    let plane = Plane::new(p3(f.origin), v3(f.z), v3(f.y));
    let mut at = Point2::origin();
    let mut heading = 0.0_f64;
    path.iter()
        .map(|seg| {
            let curve = match *seg {
                model::PathSeg::Line { length } => {
                    let b = at + nalgebra::Vector2::new(heading.cos(), heading.sin()) * length;
                    let c = Curve2::Line {
                        a: at,
                        b,
                        source: None,
                    };
                    at = b;
                    c
                }
                model::PathSeg::Arc { radius, angle } => {
                    let side = angle.signum();
                    let to_center = heading + side * FRAC_PI_2;
                    let center =
                        at + nalgebra::Vector2::new(to_center.cos(), to_center.sin()) * radius;
                    let start_angle = to_center + std::f64::consts::PI;
                    let end = start_angle + angle;
                    at = center + nalgebra::Vector2::new(end.cos(), end.sin()) * radius;
                    heading += angle;
                    Curve2::Arc {
                        center,
                        radius,
                        start_angle,
                        sweep: angle,
                        source: None,
                    }
                }
            };
            PathCurve::Sketch { plane, curve }
        })
        .collect()
}

/// The suite's B-spline path, its control points in the frame's coordinates, in the world's.
pub fn spline_path(f: &model::Frame, path: &brep_protocol::path::PathCurve) -> PathCurve {
    PathCurve::Bspline(Bspline3 {
        degree: path.degree,
        knots: path.knots.clone(),
        poles: path.ctrl.iter().map(|&c| to_world(f, c)).collect(),
        weights: Vec::new(),
    })
}
