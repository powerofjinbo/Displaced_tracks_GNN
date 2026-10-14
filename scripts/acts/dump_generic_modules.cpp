// Dump every sensitive surface of the ACTS v15 GenericDetector with what the TrackML detectors.csv records per module:
// geometry id, centre, rotation (columns u, v, w), bounds in the TrackML convention (module_minhu at v = -hv,
// module_maxhu at v = +hv, module_hv), thickness and the readout pitch of the element's digitisation module. Also
// writes the element identifier (the builder's running module counter), which shows the build order behind the
// sensitive ids. Used by gate G0 of docs/pileup_overlay_plan.md (section 10) for the z-mirror patched build and, as a
// control, for the unpatched build-v15.
//
// Build and run inside the v15 container against a build tree (slurm/acts/dump_generic_modules.sbatch):
//     dump_generic_modules OUT.csv
// No ACTS file is changed; the tool is compiled outside the source and build trees.

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Plugins/Digitization/CartesianSegmentation.hpp"
#include "Acts/Plugins/Digitization/DigitizationModule.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/TrapezoidBounds.hpp"
#include "Acts/Utilities/BinUtility.hpp"
#include "ActsExamples/GenericDetector/GenericDetector.hpp"
#include "ActsExamples/GenericDetector/GenericDetectorElement.hpp"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: dump_generic_modules OUT.csv\n";
    return 2;
  }
  GenericDetector detector;  // owns the detector elements the surfaces point to
  GenericDetector::Config cfg;
  cfg.surfaceLogLevel = Acts::Logging::WARNING;
  cfg.layerLogLevel = Acts::Logging::WARNING;
  cfg.volumeLogLevel = Acts::Logging::WARNING;
  auto [geometry, decorators] = detector.finalize(cfg, nullptr);
  Acts::GeometryContext gctx;

  std::ofstream out(argv[1]);
  out << "geometry_id,volume_id,layer_id,module_id,element_id,cx,cy,cz,"
         "rot_xu,rot_xv,rot_xw,rot_yu,rot_yv,rot_yw,rot_zu,rot_zv,rot_zw,"
         "bounds,module_minhu,module_maxhu,module_hv,module_t,"
         "half_thickness_digi,readout_direction,lorentz_angle,bins_u,bins_v,"
         "seg_min_u,seg_max_u,seg_min_v,seg_max_v,pitch_u,pitch_v\n";
  out << std::setprecision(17);
  size_t nSensitive = 0, nOther = 0, nBad = 0;
  geometry->visitSurfaces([&](const Acts::Surface* s) {
    const Acts::GeometryIdentifier gid = s->geometryId();
    if (gid.sensitive() == 0) {
      ++nOther;
      return;
    }
    const auto* elem = dynamic_cast<const ActsExamples::Generic::GenericDetectorElement*>(
        s->associatedDetectorElement());
    if (elem == nullptr) {
      ++nBad;
      return;
    }
    ++nSensitive;
    const Acts::Transform3& tf = s->transform(gctx);
    const Acts::Vector3 c = tf.translation();
    const Acts::RotationMatrix3 r = tf.linear();  // the stored matrix, no polar decomposition
    std::string btype;
    double minhu = 0., maxhu = 0., hv = 0.;
    if (const auto* tb = dynamic_cast<const Acts::TrapezoidBounds*>(&s->bounds())) {
      btype = "trapezoid";
      minhu = tb->get(Acts::TrapezoidBounds::eHalfLengthXnegY);
      maxhu = tb->get(Acts::TrapezoidBounds::eHalfLengthXposY);
      hv = tb->get(Acts::TrapezoidBounds::eHalfLengthY);
    } else if (const auto* rb = dynamic_cast<const Acts::RectangleBounds*>(&s->bounds())) {
      btype = "rectangle";
      // the GenericDetector rectangles are centred; record the half lengths and flag any offset below
      minhu = maxhu = rb->halfLengthX();
      hv = rb->halfLengthY();
      if (rb->get(Acts::RectangleBounds::eMinX) != -rb->get(Acts::RectangleBounds::eMaxX) ||
          rb->get(Acts::RectangleBounds::eMinY) != -rb->get(Acts::RectangleBounds::eMaxY)) {
        btype = "rectangle_offset";
      }
    } else {
      btype = "other";
    }
    double hthick = -1., lorentz = 0.;
    int readout = 0;
    size_t binsU = 0, binsV = 0;
    double segMinU = 0., segMaxU = 0., segMinV = 0., segMaxV = 0.;
    const auto dm = elem->digitizationModule();
    if (dm) {
      hthick = dm->halfThickness();
      readout = dm->readoutDirection();
      lorentz = dm->lorentzAngle();
      const auto* seg = dynamic_cast<const Acts::CartesianSegmentation*>(&dm->segmentation());
      if (seg != nullptr) {
        const auto& bd = seg->binUtility().binningData();
        binsU = bd.at(0).bins();
        segMinU = bd.at(0).min;
        segMaxU = bd.at(0).max;
        if (bd.size() > 1) {
          binsV = bd.at(1).bins();
          segMinV = bd.at(1).min;
          segMaxV = bd.at(1).max;
        }
      }
    }
    const double pitchU = binsU ? (segMaxU - segMinU) / binsU : 0.;
    const double pitchV = binsV ? (segMaxV - segMinV) / binsV : 0.;
    out << gid.value() << ',' << gid.volume() << ',' << gid.layer() << ',' << gid.sensitive() << ','
        << elem->identifier() << ',' << c.x() << ',' << c.y() << ',' << c.z();
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        out << ',' << r(row, col);
      }
    }
    out << ',' << btype << ',' << minhu << ',' << maxhu << ',' << hv << ',' << elem->thickness() << ','
        << hthick << ',' << readout << ',' << lorentz << ',' << binsU << ',' << binsV << ',' << segMinU << ','
        << segMaxU << ',' << segMinV << ',' << segMaxV << ',' << pitchU << ',' << pitchV << '\n';
  });
  out.close();
  std::cout << "sensitive " << nSensitive << " non_sensitive_visited " << nOther << " without_generic_element "
            << nBad << " -> " << argv[1] << std::endl;
  return nBad == 0 ? 0 : 1;
}
