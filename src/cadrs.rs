//! The B-rep protocol's functions, each answered by cadrs's one call for it.

use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::{Mutex, MutexGuard};

use brep_protocol::{Kernel as Protocol, Refusal, model, model::Primitive, profile};
use cadrs_kernel::native::{NativeKernel, tol};
use cadrs_kernel::{
    Axis, BodyId, BoolOp, ChamferOpts, ChamferSpec, DistanceTarget, DraftSpec, EdgeId, Extent,
    FaceId, FaceInput, FilletLaw, FilletSpec, Helix, Kernel, KernelError, LoftSection, LoftSpec,
    Motion, OffsetSpec, OpResult, Pick, PointClass, ProjVisibility, ProjectOptions, ShellSpec,
    SplitTool, SweepControl, SweepLaw, SweepSpec, Tessellation, ThickenSpec, ViewFrame,
};
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};

use crate::sketch::{legs, meridian, p3, sketch, spline_path, to_world, v3, xy_plane};

/// The cadrs_kernel release the adapter is built on: the `rev` in Cargo.toml.
const VERSION: &str = "0.2.8 (6cae1559)";

/// A solid, or `None` for the empty solid: cadrs has no empty body, and says so by
/// `EmptyResult`.
pub type Solid = Option<BodyId>;

/// One cadrs session. Its operations never change their inputs, so a body id stays the solid it
/// named.
pub struct Cadrs(Mutex<NativeKernel>);

/// cadrs's error as the protocol's refusal, its kind the kernel's own.
fn refusal(e: KernelError) -> Refusal {
    let reason = e.to_string();
    match e {
        KernelError::Unsupported(_) => Refusal::unsupported(reason),
        KernelError::InvalidProfile(_)
        | KernelError::InvalidParameter(_)
        | KernelError::InvalidInput(_)
        | KernelError::UnknownBody(_) => Refusal::invalid(reason),
        _ => Refusal::failed(reason),
    }
}

/// The one body an operation makes.
fn one(r: cadrs_kernel::Result<OpResult>) -> Result<Solid, Refusal> {
    r.and_then(|r| r.first()).map(Some).map_err(refusal)
}

/// A boolean's body, or the empty solid where cadrs says nothing is left.
fn empty_or(r: cadrs_kernel::Result<OpResult>) -> Result<Solid, Refusal> {
    match r {
        Err(KernelError::EmptyResult(_)) => Ok(None),
        r => one(r),
    }
}

fn solid(s: &Solid) -> Result<BodyId, Refusal> {
    s.ok_or_else(|| Refusal::invalid("the solid is empty"))
}

fn linear(m: &[[f64; 4]; 3]) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| m[i][j])
}

fn shift(m: &[[f64; 4]; 3]) -> Vector3<f64> {
    Vector3::new(m[0][3], m[1][3], m[2][3])
}

/// The rigid motion taking the world's axes onto the frame's.
fn placement(f: &model::Frame) -> Isometry3<f64> {
    let r = Matrix3::from_columns(&[v3(f.x), v3(f.y), v3(f.z)]);
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    Isometry3::from_parts(Translation3::from(v3(f.origin)), q)
}

impl Cadrs {
    pub fn new() -> Self {
        Cadrs(Mutex::new(NativeKernel::new()))
    }

    fn k(&self) -> MutexGuard<'_, NativeKernel> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The contract's `10·ε`, within which a point names an edge or a face.
const NEAR: f64 = 10.0 * tol::POINT;

/// The edge each point lies on: the kernel's nearest edge, which must pass within `10·ε`.
fn edges_at(k: &NativeKernel, b: BodyId, points: &[model::Vec3]) -> Result<Vec<EdgeId>, Refusal> {
    points
        .iter()
        .map(|&p| {
            let e = k.edge_at(b, p3(p)).map_err(refusal)?;
            let d = k
                .min_distance(&DistanceTarget::Point(p3(p)), &DistanceTarget::Edge(b, e))
                .map_err(refusal)?;
            if d.distance <= NEAR {
                Ok(e)
            } else {
                Err(Refusal::invalid(format!("no edge passes through {p:?}")))
            }
        })
        .collect()
}

/// The face a point lies on: the kernel's nearest face, which must pass within `10·ε`.
fn face_at(k: &NativeKernel, b: BodyId, p: model::Vec3) -> Result<Option<FaceId>, Refusal> {
    let f = k.face_at(b, p3(p)).map_err(refusal)?;
    let d = k
        .min_distance(&DistanceTarget::Point(p3(p)), &DistanceTarget::Face(b, f))
        .map_err(refusal)?;
    Ok((d.distance <= NEAR).then_some(f))
}

fn faces_at(k: &NativeKernel, b: BodyId, points: &[model::Vec3]) -> Result<Vec<FaceId>, Refusal> {
    points
        .iter()
        .map(|&p| {
            face_at(k, b, p)?
                .ok_or_else(|| Refusal::invalid(format!("no face passes through {p:?}")))
        })
        .collect()
}

impl Protocol for Cadrs {
    type Solid = Solid;

    fn name(&self) -> String {
        "cadrs".into()
    }

    fn version(&self) -> String {
        VERSION.into()
    }

    fn tolerance(&self) -> f64 {
        tol::POINT
    }

    fn reset(&self) {
        *self.k() = NativeKernel::new();
    }

    fn unsupported(&self) -> Vec<String> {
        [
            // Functions cadrs does not have.
            "slab",
            "fillet_g2",
            "fillet_setback_g2",
            "curvature",
            "betti",
            "section",
            "section_area",
            "interference",
            // cadrs sizes a conic fillet by the radius of the circular fillet it replaces; the
            // protocol sizes it by its setback from the edge.
            "fillet_conic",
            // cadrs's face blend needs two faces that share an edge, and makes a sheet.
            "fillet_faces",
            // cadrs measures distances between boundaries; the protocol counts a solid's inside.
            "closest",
            "distance",
        ]
        .map(String::from)
        .to_vec()
    }

    fn make(&self, p: &Primitive) -> Result<Solid, Refusal> {
        let f = p.frame();
        let (o, z) = (p3(f.origin), nalgebra::Unit::new_normalize(v3(f.z)));
        let k = &mut *self.k();
        match *p {
            Primitive::Box { size, .. } => {
                // cadrs builds a box square to the world's axes; it is moved onto the frame.
                let min = nalgebra::Point3::origin();
                let made = one(k.primitive(&cadrs_kernel::Primitive::Box { min, max: p3(size) }))?;
                one(k.transform(solid(&made)?, &placement(&f)))
            }
            Primitive::Cylinder { radius, height, .. } => {
                one(k.primitive(&cadrs_kernel::Primitive::Cylinder {
                    base: o,
                    axis: z,
                    radius,
                    height,
                }))
            }
            Primitive::Cone {
                bottom,
                top,
                height,
                ..
            } => one(k.primitive(&cadrs_kernel::Primitive::Cone {
                base: o,
                axis: z,
                base_radius: bottom,
                top_radius: top,
                height,
            })),
            Primitive::Sphere { radius, .. } => {
                one(k.primitive(&cadrs_kernel::Primitive::Sphere {
                    center: o,
                    radius,
                    axis: z,
                    seam: nalgebra::Unit::new_normalize(v3(f.x)),
                }))
            }
            Primitive::Torus { major, minor, .. } => {
                one(k.primitive(&cadrs_kernel::Primitive::Torus {
                    center: o,
                    axis: z,
                    major,
                    minor,
                }))
            }
            Primitive::Extrude {
                ref profile,
                height,
                ..
            } => one(k.extrude(&sketch(profile, xy_plane(&f))?, Extent::Blind(height))),
            Primitive::Revolve {
                ref profile, angle, ..
            } => one(k.revolve(
                &sketch(profile, meridian(&f))?,
                Axis { origin: o, dir: z },
                angle,
            )),
            Primitive::Screw {
                ref profile,
                rise,
                angle,
                ..
            } => {
                // The helix the profile's screw carries it along, through the middle of the
                // profile's extent out from the axis.
                let ([x0, _], [x1, _]) = profile.bounds();
                let helix = Helix {
                    axis: Axis { origin: o, dir: z },
                    x: nalgebra::Unit::new_normalize(v3(f.x)),
                    r0: 0.5 * (x0 + x1),
                    r1: 0.5 * (x0 + x1),
                    height: rise * angle,
                    turns: angle / TAU,
                    start_angle: 0.0,
                };
                let mut spec = SweepSpec::solid(helix.path());
                spec.control = SweepControl::Helix(helix);
                one(k.sweep_with(&sketch(profile, meridian(&f))?, &spec))
            }
            Primitive::Sweep {
                ref profile,
                ref path,
                ..
            } => {
                let spec = SweepSpec::solid(legs(&f, path));
                one(k.sweep_with(&sketch(profile, xy_plane(&f))?, &spec))
            }
            Primitive::SweepPath {
                ref profile,
                ref path,
                twist,
                ..
            } => {
                let spec = SweepSpec::solid(vec![spline_path(&f, path)]);
                let law = SweepLaw {
                    twist,
                    ..SweepLaw::default()
                };
                one(k.sweep_with_law(&sketch(profile, xy_plane(&f))?, &spec, &law))
            }
            Primitive::Slab { .. } => Err(Refusal::unsupported("cadrs has no height-field slab")),
            Primitive::Loft {
                ref profile,
                scale,
                height,
                ..
            } => {
                // The top section is the protocol crate's scaled copy, the same for every adapter.
                let lifted = to_world(&f, [0.0, 0.0, height]);
                let top = model::Frame {
                    origin: [lifted.x, lifted.y, lifted.z],
                    ..f
                };
                let sections = vec![
                    LoftSection::Profile(sketch(profile, xy_plane(&f))?),
                    LoftSection::Profile(sketch(&profile.scaled(scale), xy_plane(&top))?),
                ];
                one(k.loft_with(&LoftSpec::solid(sections)))
            }
        }
    }

    fn union(&self, a: &Solid, b: &Solid) -> Result<Solid, Refusal> {
        match (a, b) {
            (Some(a), Some(b)) => empty_or(self.k().boolean(BoolOp::Union, *a, &[*b])),
            (x, None) | (None, x) => Ok(*x),
        }
    }

    fn intersection(&self, a: &Solid, b: &Solid) -> Result<Solid, Refusal> {
        match (a, b) {
            (Some(a), Some(b)) => empty_or(self.k().boolean(BoolOp::Intersect, *a, &[*b])),
            _ => Ok(None),
        }
    }

    fn difference(&self, a: &Solid, b: &Solid) -> Result<Solid, Refusal> {
        match (a, b) {
            (Some(a), Some(b)) => empty_or(self.k().boolean(BoolOp::Subtract, *a, &[*b])),
            (x, None) => Ok(*x),
            (None, _) => Ok(None),
        }
    }

    fn union_all(&self, solids: &[&Solid]) -> Result<Solid, Refusal> {
        let mut bodies = solids.iter().filter_map(|s| **s);
        let Some(first) = bodies.next() else {
            return Ok(None);
        };
        let rest: Vec<BodyId> = bodies.collect();
        if rest.is_empty() {
            return Ok(Some(first));
        }
        empty_or(self.k().boolean(BoolOp::Union, first, &rest))
    }

    fn difference_all(&self, target: &Solid, tools: &[&Solid]) -> Result<Solid, Refusal> {
        let Some(target) = *target else {
            return Ok(None);
        };
        let tools: Vec<BodyId> = tools.iter().filter_map(|s| **s).collect();
        if tools.is_empty() {
            return Ok(Some(target));
        }
        empty_or(self.k().boolean(BoolOp::Subtract, target, &tools))
    }

    fn volume(&self, s: &Solid) -> Result<f64, Refusal> {
        match s {
            None => Ok(0.0),
            Some(b) => Ok(self.k().mass_properties(*b).map_err(refusal)?.volume),
        }
    }

    fn area(&self, s: &Solid) -> Result<f64, Refusal> {
        match s {
            None => Ok(0.0),
            Some(b) => Ok(self.k().mass_properties(*b).map_err(refusal)?.surface_area),
        }
    }

    fn check(&self, s: &Solid) -> Result<Vec<String>, Refusal> {
        match s {
            None => Ok(Vec::new()),
            Some(b) => Ok(self.k().get(*b).map_err(refusal)?.check()),
        }
    }

    fn classify(&self, s: &Solid, points: &[model::Vec3]) -> Result<Vec<model::Side>, Refusal> {
        let Some(b) = *s else {
            return Ok(vec![model::Side::Out; points.len()]);
        };
        let k = self.k();
        points
            .iter()
            .map(|&p| {
                Ok(match k.classify(b, p3(p), tol::POINT).map_err(refusal)? {
                    PointClass::Inside => model::Side::In,
                    PointClass::OnBoundary => model::Side::On,
                    PointClass::Outside => model::Side::Out,
                })
            })
            .collect()
    }

    fn normal(
        &self,
        s: &Solid,
        points: &[model::Vec3],
    ) -> Result<Vec<Option<model::Vec3>>, Refusal> {
        let Some(b) = *s else {
            return Ok(vec![None; points.len()]);
        };
        let k = self.k();
        points
            .iter()
            .map(|&p| {
                // A point on an edge lies on two faces, and cadrs answers on one face.
                if edges_at(&k, b, &[p]).is_ok() {
                    return Err(Refusal::unsupported(
                        "cadrs gives a face's normal; this point lies on an edge",
                    ));
                }
                let Some(f) = face_at(&k, b, p)? else {
                    return Ok(None);
                };
                let n = k.face_normals_at(b, f, &[p3(p)]).map_err(refusal)?;
                Ok(n.first().map(|n| [n.x, n.y, n.z]))
            })
            .collect()
    }

    fn fillet(&self, s: &Solid, edges: &[model::Vec3], radius: f64) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let ids = edges_at(k, b, edges)?;
        one(k.fillet(b, &ids, radius))
    }

    fn fillet_radii(
        &self,
        s: &Solid,
        edges: &[model::Vec3],
        radii: &[f64],
    ) -> Result<Solid, Refusal> {
        if edges.len() != radii.len() {
            return Err(Refusal::invalid(format!(
                "{} edges but {} radii",
                edges.len(),
                radii.len()
            )));
        }
        let b = solid(s)?;
        let k = &mut *self.k();
        let laws: Vec<FilletLaw> = edges_at(k, b, edges)?
            .into_iter()
            .zip(radii)
            .map(|(edge, &r)| FilletLaw {
                edge,
                radii: vec![(0.0, r), (1.0, r)],
            })
            .collect();
        one(k.fillet_variable(b, &laws, true))
    }

    fn fillet_variable(
        &self,
        s: &Solid,
        edge: model::Vec3,
        from: model::Vec3,
        r_from: f64,
        r_to: f64,
    ) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let e = edges_at(k, b, &[edge])?[0];
        // cadrs gives the radii from the edge's start; `from` names one of its ends.
        let info = k.edges(b).map_err(refusal)?;
        let info = info
            .iter()
            .find(|i| i.id == e)
            .ok_or_else(|| Refusal::failed("the picked edge is not among the body's edges"))?;
        let from = p3(from);
        let radii = if (info.start - from).norm() <= (info.end - from).norm() {
            vec![(0.0, r_from), (1.0, r_to)]
        } else {
            vec![(0.0, r_to), (1.0, r_from)]
        };
        one(k.fillet_variable(b, &[FilletLaw { edge: e, radii }], true))
    }

    fn fillet_setback(
        &self,
        s: &Solid,
        edges: &[model::Vec3],
        radius: f64,
        setback: f64,
    ) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let ids = edges_at(k, b, edges)?;
        one(k.fillet_smooth(b, &ids, &FilletSpec::radius(radius), setback))
    }

    fn full_round(
        &self,
        s: &Solid,
        side_a: model::Vec3,
        middle: model::Vec3,
        side_b: model::Vec3,
    ) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let f = faces_at(k, b, &[side_a, middle, side_b])?;
        one(k.full_round(b, f[0], f[1], f[2]))
    }

    fn chamfer(&self, s: &Solid, edges: &[model::Vec3], distance: f64) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let ids = edges_at(k, b, edges)?;
        one(k.chamfer(b, &ids, ChamferSpec::EqualDistance(distance)))
    }

    fn chamfer_two(
        &self,
        s: &Solid,
        edge: model::Vec3,
        face: model::Vec3,
        d1: f64,
        d2: f64,
    ) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let e = edges_at(k, b, &[edge])?;
        let f = faces_at(k, b, &[face])?[0];
        let opts = ChamferOpts::new(ChamferSpec::TwoDistances(d1, d2)).first_on(f);
        one(k.chamfer_with(b, &e, &opts))
    }

    fn chamfer_distances(
        &self,
        s: &Solid,
        edges: &[model::Vec3],
        distances: &[f64],
    ) -> Result<Solid, Refusal> {
        if edges.len() != distances.len() {
            return Err(Refusal::invalid(format!(
                "{} edges but {} sizes",
                edges.len(),
                distances.len()
            )));
        }
        let Some(&d) = distances.first() else {
            return Err(Refusal::invalid("no edges"));
        };
        if distances.iter().any(|&x| x != d) {
            return Err(Refusal::unsupported(
                "cadrs's chamfer takes one distance for all its edges",
            ));
        }
        self.chamfer(s, edges, d)
    }

    fn offset(&self, s: &Solid, distance: f64) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let spec = OffsetSpec {
            distance,
            faces: Vec::new(),
            sharp: false,
        };
        one(self.k().offset(b, &spec))
    }

    fn offset_faces(
        &self,
        s: &Solid,
        faces: &[model::Vec3],
        distance: f64,
    ) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let by: HashMap<FaceId, f64> = faces_at(k, b, faces)?
            .into_iter()
            .map(|f| (f, distance))
            .collect();
        let spec = OffsetSpec {
            distance: 0.0,
            faces: by.into_iter().collect(),
            sharp: true,
        };
        one(k.offset(b, &spec))
    }

    fn delete_faces(&self, s: &Solid, faces: &[model::Vec3]) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let faces = faces_at(k, b, faces)?;
        one(k.delete_faces(b, &faces))
    }

    fn move_faces(
        &self,
        s: &Solid,
        faces: &[model::Vec3],
        matrix: [[f64; 4]; 3],
    ) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let faces = faces_at(k, b, faces)?;
        let motion = Motion {
            linear: linear(&matrix),
            translation: shift(&matrix),
        };
        one(k.move_faces(b, &faces, &motion))
    }

    fn thicken(&self, s: &Solid, faces: &[model::Vec3], thickness: f64) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let faces = faces_at(k, b, faces)?
            .into_iter()
            .map(|face| FaceInput {
                body: b,
                face,
                source: 0,
            })
            .collect();
        let spec = ThickenSpec {
            bodies: Vec::new(),
            faces,
            profiles: Vec::new(),
            along: thickness.max(0.0),
            against: (-thickness).max(0.0),
            sharp: false,
            source: 0,
        };
        one(k.thicken_surfaces(&spec))
    }

    fn shell(&self, s: &Solid, open: &[model::Vec3], thickness: f64) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let spec = ShellSpec {
            remove: faces_at(k, b, open)?,
            thickness,
            outward: false,
            // With no face open, the protocol's shell is a closed hollow.
            hollow: open.is_empty(),
            sharp: false,
        };
        one(k.shell_with(b, &spec))
    }

    fn draft(
        &self,
        s: &Solid,
        faces: &[model::Vec3],
        origin: model::Vec3,
        pull: model::Vec3,
        angle: f64,
    ) -> Result<Solid, Refusal> {
        let b = solid(s)?;
        let k = &mut *self.k();
        let spec = DraftSpec {
            faces: faces_at(k, b, faces)?,
            angle,
            pull: nalgebra::Unit::new_normalize(v3(pull)),
            neutral: p3(origin),
            neutral_normal: None,
            tangent_propagation: false,
        };
        one(k.draft(b, &spec))
    }

    fn write_step(&self, s: &Solid) -> Result<String, Refusal> {
        let bodies: Vec<BodyId> = s.iter().copied().collect();
        let bytes = self.k().export_step(&bodies).map_err(refusal)?;
        String::from_utf8(bytes).map_err(|e| Refusal::failed(e.to_string()))
    }

    fn read_step(&self, text: &str) -> Result<Solid, Refusal> {
        let k = &mut *self.k();
        let bodies = k.import_step(text.as_bytes()).map_err(refusal)?;
        // cadrs reads one body per solid; the protocol's answer is one solid set.
        match bodies.as_slice() {
            [] => Ok(None),
            [b] => Ok(Some(*b)),
            _ => one(k.compound(&bodies)),
        }
    }

    fn sew(&self, text: &str, tolerance: f64) -> Result<Solid, Refusal> {
        let k = &mut *self.k();
        let sheets = k.import_step(text.as_bytes()).map_err(refusal)?;
        one(k.sew_solid(&sheets, tolerance))
    }

    fn centroid(&self, s: &Solid) -> Result<Option<model::Vec3>, Refusal> {
        let Some(b) = *s else {
            return Ok(None);
        };
        let c = self.k().mass_properties(b).map_err(refusal)?.center_of_mass;
        Ok(Some([c.x, c.y, c.z]))
    }

    fn inertia(&self, s: &Solid) -> Result<[[f64; 3]; 3], Refusal> {
        let Some(b) = *s else {
            return Ok([[0.0; 3]; 3]);
        };
        let m = self.k().mass_properties(b).map_err(refusal)?.inertia;
        Ok([0, 1, 2].map(|i| [0, 1, 2].map(|j| m[(i, j)])))
    }

    fn loft_through(
        &self,
        frame: &model::Frame,
        profiles: &[profile::Profile],
        heights: &[f64],
    ) -> Result<Solid, Refusal> {
        if profiles.len() != heights.len() {
            return Err(Refusal::invalid(format!(
                "{} profiles at {} heights",
                profiles.len(),
                heights.len()
            )));
        }
        let sections = profiles
            .iter()
            .zip(heights)
            .map(|(p, &h)| {
                let lifted = to_world(frame, [0.0, 0.0, h]);
                let at = model::Frame {
                    origin: [lifted.x, lifted.y, lifted.z],
                    ..*frame
                };
                Ok(LoftSection::Profile(sketch(p, xy_plane(&at))?))
            })
            .collect::<Result<Vec<_>, Refusal>>()?;
        one(self.k().loft_with(&LoftSpec::solid(sections)))
    }

    fn transform(&self, s: &Solid, m: [[f64; 4]; 3]) -> Result<Solid, Refusal> {
        self.affine(s, m)
    }

    fn affine(&self, s: &Solid, m: [[f64; 4]; 3]) -> Result<Solid, Refusal> {
        let Some(b) = *s else {
            return Ok(None);
        };
        one(self.k().transform_affine(b, linear(&m), shift(&m)))
    }

    fn raycast(&self, s: &Solid, o: model::Vec3, d: model::Vec3) -> Result<Vec<f64>, Refusal> {
        let Some(b) = *s else {
            return Ok(Vec::new());
        };
        // cadrs fires a line along a unit direction; the ray is its part ahead of the origin.
        let hits = self
            .k()
            .ray_hits(b, p3(o), v3(d).normalize())
            .map_err(refusal)?;
        Ok(hits.iter().map(|h| h.t).filter(|&t| t >= 0.0).collect())
    }

    fn mesh(&self, s: &Solid, tolerance: f64) -> Result<brep_protocol::Mesh, Refusal> {
        let Some(b) = *s else {
            return Ok(brep_protocol::Mesh {
                positions: Vec::new(),
                triangles: Vec::new(),
            });
        };
        let quality = Tessellation {
            deflection: tolerance,
            ..Tessellation::default()
        };
        let m = self.k().tessellate(b, quality).map_err(refusal)?;
        Ok(brep_protocol::Mesh {
            positions: m.positions.iter().map(|p| [p.x, p.y, p.z]).collect(),
            triangles: m.indices,
        })
    }

    fn display(&self, s: &Solid, tolerance: f64) -> Result<brep_protocol::Display, Refusal> {
        let Some(b) = *s else {
            return Ok(brep_protocol::Display::default());
        };
        let k = self.k();
        let quality = Tessellation {
            deflection: tolerance,
            ..Tessellation::default()
        };
        let m = k.tessellate(b, quality).map_err(refusal)?;
        let creases: Vec<EdgeId> = k
            .edges(b)
            .map_err(refusal)?
            .into_iter()
            // A crease is where two faces meet at an angle: not a seam, and not an edge of no face.
            .filter(|e| match e.faces {
                [Some(a), Some(b)] => a != b && !e.smooth,
                _ => false,
            })
            .map(|e| e.id)
            .collect();
        // cadrs tags each triangle with its face; the protocol gives each face its own vertices.
        let mut faces: Vec<(FaceId, brep_protocol::DisplayFace, HashMap<u32, u32>)> = Vec::new();
        for (t, f) in m.indices.iter().zip(&m.triangle_faces) {
            let at = match faces.iter().position(|(id, ..)| id == f) {
                Some(at) => at,
                None => {
                    faces.push((*f, brep_protocol::DisplayFace::default(), HashMap::new()));
                    faces.len() - 1
                }
            };
            let (_, face, index) = &mut faces[at];
            let tri = t.map(|i| {
                *index.entry(i).or_insert_with(|| {
                    let (p, n) = (m.positions[i as usize], m.normals[i as usize]);
                    face.positions.push([p.x, p.y, p.z]);
                    face.normals.push([n.x, n.y, n.z]);
                    (face.positions.len() - 1) as u32
                })
            });
            face.triangles.push(tri);
        }
        Ok(brep_protocol::Display {
            faces: faces.into_iter().map(|(_, f, _)| f).collect(),
            edges: m
                .edges
                .iter()
                .filter(|(id, _)| creases.contains(id))
                .map(|(_, line)| line.iter().map(|p| [p.x, p.y, p.z]).collect())
                .collect(),
        })
    }

    fn split(
        &self,
        s: &Solid,
        origin: model::Vec3,
        normal: model::Vec3,
    ) -> Result<(Solid, Solid), Refusal> {
        let Some(b) = *s else {
            return Ok((None, None));
        };
        let k = &mut *self.k();
        let n = v3(normal);
        let x = if n.x.abs() < 0.9 * n.norm() {
            Vector3::x()
        } else {
            Vector3::y()
        };
        let plane = cadrs_kernel::Plane::new(p3(origin), x, n);
        let cut = one(k.split(b, &SplitTool::Plane(plane), 0))?;
        let pieces = k.split_solids(solid(&cut)?).map_err(refusal)?;
        // cadrs gives one body of all the pieces; the protocol sorts them by side.
        let (mut front, mut back) = (Vec::new(), Vec::new());
        for piece in pieces {
            let piece = piece.first().map_err(refusal)?;
            let c = k.mass_properties(piece).map_err(refusal)?.center_of_mass;
            if (c - p3(origin)).dot(&n) >= 0.0 {
                front.push(piece);
            } else {
                back.push(piece);
            }
        }
        let mut gather = |bodies: Vec<BodyId>| match bodies.as_slice() {
            [] => Ok(None),
            [b] => Ok(Some(*b)),
            _ => one(k.compound(&bodies)),
        };
        Ok((gather(front)?, gather(back)?))
    }

    fn project(
        &self,
        s: &Solid,
        direction: model::Vec3,
        tolerance: f64,
    ) -> Result<brep_protocol::Projection, Refusal> {
        let Some(b) = *s else {
            return Ok(brep_protocol::Projection {
                visible: Vec::new(),
                hidden: Vec::new(),
            });
        };
        let (u, v) = brep_protocol::view_axes(direction);
        let view = ViewFrame::new(v3(direction), v3(u));
        let opts = ProjectOptions {
            tolerance,
            hidden: true,
        };
        let drawn = self.k().project(&[b], &view, &opts).map_err(refusal)?;
        // cadrs's 2D y is `x × dir`; the protocol's is `v`.
        let (x, up, v) = (view.x, view.up(), v3(v));
        let to_uv = |p: &nalgebra::Point2<f64>| {
            let w = x * p.x + up * p.y;
            [w.dot(&v3(u)), w.dot(&v)]
        };
        let (mut visible, mut hidden) = (Vec::new(), Vec::new());
        for e in &drawn.edges {
            let line = e.points.iter().map(to_uv).collect();
            match e.visibility {
                ProjVisibility::Visible => visible.push(line),
                ProjVisibility::Hidden => hidden.push(line),
            }
        }
        Ok(brep_protocol::Projection { visible, hidden })
    }

    fn bounding_box(&self, s: &Solid) -> Result<[model::Vec3; 2], Refusal> {
        let b = solid(s)?;
        let bb = self.k().bounding_box(b).map_err(refusal)?;
        Ok([
            [bb.min.x, bb.min.y, bb.min.z],
            [bb.max.x, bb.max.y, bb.max.z],
        ])
    }
}
