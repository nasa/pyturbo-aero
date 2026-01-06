import numpy as np

from pyturbo.aero import Centrif, CentrifProfile, TrailingEdgeProperties


def _build_simple_centrif() -> Centrif:
    xhub = np.linspace(0.0, 1.0, 50)
    rhub = 1.0 + 0.05 * np.sin(np.linspace(0.0, np.pi, 50))
    xshroud = xhub.copy()
    rshroud = rhub + 0.5

    cen = Centrif(blade_position=(0.0, 1.0), use_mid_wrap_angle=True, use_ray_camber=False)
    cen.add_hub(xhub, rhub)
    cen.add_shroud(xshroud, rshroud)

    te_props = TrailingEdgeProperties(TE_Cut=False, TE_Radius=0.01)

    def prof(span: float) -> CentrifProfile:
        return CentrifProfile(
            percent_span=span,
            LE_Thickness=0.02,
            LE_Metal_Angle=0.0,
            TE_Metal_Angle=60.0,
            LE_Metal_Angle_Loc=0.1,
            TE_Metal_Angle_Loc=0.9,
            ss_thickness=[0.04, 0.03, 0.03, 0.03],
            ps_thickness=[0.02, 0.03, 0.03, 0.03],
            wrap_angle=-20.0,
            wrap_displacements=[0.0, 0.0],
            wrap_displacement_locs=[0.4, 0.8],
            trailing_edge_properties=te_props,
        )

    cen.add_profile(prof(0.0))
    cen.add_profile(prof(0.5))
    cen.add_profile(prof(1.0))

    cen.build(npts_span=7, npts_chord=150, nblades=3, nsplitters=0)
    return cen


def test_passage_offset_matches_boundaries():
    cen = _build_simple_centrif()

    xr, hub_xr, shroud_xr, t = cen.get_passage_xr_streamline_by_span_offset(0.0, from_="hub")
    assert xr.shape == hub_xr.shape
    assert np.allclose(xr, hub_xr)
    assert np.allclose(t, 0.0)

    xr, hub_xr, shroud_xr, t = cen.get_passage_xr_streamline_by_span_offset(0.0, from_="shroud")
    assert np.allclose(xr, shroud_xr)
    assert np.allclose(t, 1.0)


def test_blade_offset_matches_built_span_rows():
    cen = _build_simple_centrif()

    ss, ps, t = cen.get_blade_cyl_streamline_by_span_offset(0.0, from_="hub")
    assert ss.shape == (cen.npts_chord, 3)
    assert ps.shape == (cen.npts_chord, 3)
    assert np.allclose(t, 0.0)
    assert np.allclose(ss, cen.mainblade.ss_cyl_pts[0, :, :])
    assert np.allclose(ps, cen.mainblade.ps_cyl_pts[0, :, :])

    ss, ps, t = cen.get_blade_cyl_streamline_by_span_offset(0.0, from_="shroud")
    assert np.allclose(ss, cen.mainblade.ss_cyl_pts[-1, :, :])
    assert np.allclose(ps, cen.mainblade.ps_cyl_pts[-1, :, :])


def test_offset_produces_expected_tspan_for_constant_height_passage():
    cen = _build_simple_centrif()

    # passage height is ~0.5 everywhere for this setup -> offset 0.1 => tspan ~0.2
    xr, _, _, t = cen.get_passage_xr_streamline_by_span_offset(0.1, from_="hub")
    assert np.allclose(t, 0.2, atol=1e-6)

    ss, ps, t_blade = cen.get_blade_cyl_streamline_by_span_offset(0.1, from_="hub")
    assert ss.shape == (cen.npts_chord, 3)
    assert ps.shape == (cen.npts_chord, 3)
    assert np.allclose(t_blade, 0.2, atol=1e-6)
