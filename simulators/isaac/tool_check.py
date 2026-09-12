"""Tool check — prove the gripper+cutter tool and the mid-run body split in Isaac.

    ISAAC_TASK_MARKER=TOOL-CHECK-DONE ./scripts/run_isaac_task.sh \\
        /tmp/tool_check.log 900 simulators/isaac/tool_check.py

WHAT IT MEASURES, with the real Panda articulation and the real WISEPACK scene:

  1. the combined tool is built under the hand as VISUAL geometry (no physics
     schema on any of its prims), rides with the hand, and its blades close and
     open on command, measured as the world distance between the blade prims;
  2. the tool has NO CONTACT EFFECT: gravity-free tubes spawned overlapping a
     closed blade and the bracket stay exactly where they were spawned, blade
     stroke or not (the earlier rigid-body tool with collision "disabled"
     pushed a tube 54 mm — that measurement is why the tool is visual-only,
     see tool.py);
  3. two rigid segment bodies can be CREATED while physics is playing, and the
     arm's joints and Jacobian remain readable and commandable afterwards;
  4. an existing tube body can be DEACTIVATED while playing (hidden, collision
     off, parked below the table) without touching the arm;
  5. a freshly created body can be welded to the hand and released, and the
     weld reports the held item's offset in the hand frame;
  6. (last, expected to break the views) deleting a rigid prim while playing —
     recorded so the cut skill never does it.

Every check prints a measured PASS/FAIL line; the process never asserts, so one
failure does not hide the next result.
"""

from __future__ import annotations

import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        return _run(app)
    except BaseException:                                    # noqa: BLE001
        print("TOOL-CHECK FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return 1
    finally:
        try:
            app.close()
        except Exception:                                    # noqa: BLE001
            pass


def _run(app) -> int:
    import numpy as np
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import Cylinder
    from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Usd, UsdGeom, UsdPhysics

    sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))
    sys.path.insert(0, REPO)
    from wisepack_core.generator import build_scenario
    from wisepack_core.robots import load_registry
    from wisepack_core.isaac_transform import layout_for_robot
    from simulators.isaac.adapters import build_adapter
    from simulators.isaac.config import PhysicsConfig
    from simulators.isaac.grasp import GraspJoint, _quat_conjugate, _quat_rotate
    from simulators.isaac.scene import WisepackScene, item_path

    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"TOOL-CHECK [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        return ok

    def step(n):
        for _ in range(n):
            app.update()
            robot.tick_tool()

    profile = load_registry().resolve(explicit="panda")
    layout = layout_for_robot(profile)
    scene = WisepackScene(layout, PhysicsConfig())
    scenario = build_scenario("isaac_cylinders_smoke", seed=42)
    SimulationManager.set_physics_sim_device("cpu")
    app.update()
    scene.build(scenario, ["CNT-01"])

    robot = build_adapter(profile)
    robot.load(base_position=[0.0, 0.0, layout.table_top_z_m],
               base_orientation=[1.0, 0.0, 0.0, 0.0])
    app.update()

    # The adapter authored the tool under the hand BEFORE play.
    hand_path = profile.end_effector_prim
    tool = robot.tool
    stage = stage_utils.get_current_stage(backend="usd")
    schemas = {prim.GetPath().pathString: list(prim.GetAppliedSchemas())
               for prim in Usd.PrimRange(stage.GetPrimAtPath(tool.root))}
    check("tool_built_visual_only", tool.built and tool.physics_free(),
          f"root {tool.root}; applied schemas per prim {schemas}; bracket local "
          f"{tool.bracket_in_hand_m().round(4).tolist()} blade+ local "
          f"{tool.blade_rest_in_hand_m(1.0).round(4).tolist()}")

    app_utils.play()
    app.update()
    robot.initialise()
    robot.validate_model(preset="isaac_cylinders_smoke")
    robot.command_home()
    step(120)

    def arm_ok(label):
        try:
            dof = robot.get_joint_state()
            pos, ori = robot.get_tcp_pose()
            goal = np.asarray(pos) + np.array([0.0, 0.0, 0.01])
            robot.command_tcp_pose(position=goal, orientation=ori)
            step(5)
            return check(label, np.all(np.isfinite(dof)),
                         f"dof {np.round(dof[:3], 3).tolist()}..., tcp {np.round(pos, 3).tolist()}")
        except Exception as exc:                                 # noqa: BLE001
            return check(label, False, f"{type(exc).__name__}: {exc}")

    arm_ok("arm_controllable_after_play")

    # 1. The tool follows the hand and the blades actuate.
    def report(label):
        hp_, hq_ = robot.get_tcp_pose()
        parts = {}
        for name, p_ in tool.part_world_positions().items():
            local = _quat_rotate(_quat_conjugate(np.asarray(hq_)), p_ - np.asarray(hp_))
            parts[name] = np.round(local, 4).tolist()
        print(f"TOOL-DIAG {label}: tcp-frame {parts} gap {tool.blade_gap_m()}", flush=True)

    report("after_home")
    hand_now, hand_q_now = robot.get_tcp_pose()

    def measured_gap():
        parts = tool.part_world_positions()
        return float(np.linalg.norm(parts["blade_pos"] - parts["blade_neg"]))

    gap_open = measured_gap()
    tool.close_cutter()
    step(90)
    report("after_close")
    gap_closed = measured_gap()
    tool.open_cutter()
    step(90)
    report("after_reopen")
    gap_reopened = measured_gap()
    check("blades_close_and_open",
          gap_closed < 0.004 and gap_reopened > 0.065 and gap_open > 0.065,
          f"open {gap_open:.4f}, closed {gap_closed:.4f}, reopened {gap_reopened:.4f} m "
          "(world distance between the blade prims)")
    cut_world = tool.cut_frame_world(hand_now, hand_q_now)
    blades = tool.part_world_positions()
    mid = (blades["blade_pos"] + blades["blade_neg"]) / 2.0
    check("blades_ride_with_the_hand",
          float(np.linalg.norm(mid - cut_world)) < 0.01,
          f"blade midpoint {mid.round(3).tolist()} vs cut frame {cut_world.round(3).tolist()}")

    # 2. NO CONTACT EFFECT. Independent of where the arm can reach: the probe
    #    tubes are dynamic bodies with gravity DISABLED, spawned at the home
    #    pose overlapping the tool. A colliding tool would expel them on the
    #    first step (PhysX depenetration); a visual one leaves them exactly
    #    where they were spawned, blade stroke or not.
    from pxr import PhysxSchema

    def floating_tube(path, centre, colour):
        Cylinder(paths=path, radii=0.02, heights=0.15, axes="Z",
                 positions=np.array([centre]),
                 orientations=np.array([[0.7071068, 0.0, 0.7071068, 0.0]]), colors=colour)
        GeomPrim(paths=path, apply_collision_apis=True).apply_physics_materials(scene._item_material)
        body = RigidPrim(paths=path)
        body.set_masses(np.array([0.4]))
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path)).CreateDisableGravityAttr(True)
        return body

    def where(body):
        p, _ = body.get_world_poses()
        return np.asarray(p.numpy()[0], dtype=float)

    tool.close_cutter()
    step(60)
    hp_, hq_ = robot.get_tcp_pose()
    blades = tool.part_world_positions()
    hand_x = _quat_rotate(np.asarray(hq_), np.array([1.0, 0.0, 0.0]))
    # The tube lies along the hand's X, its +X end 1 mm inside the closed blade.
    cut_mid = (blades["blade_pos"] + blades["blade_neg"]) / 2.0
    probe_a = cut_mid - hand_x * (0.075 - 0.0015)
    body_a = floating_tube("/World/Items/probe_blade", probe_a, "green")
    step(60)
    pa0 = where(body_a)
    tool.open_cutter()
    step(60)
    tool.close_cutter()
    step(60)
    pa1 = where(body_a)
    moved_a = float(np.linalg.norm(pa1 - probe_a)) * 1000.0
    check("blade_has_no_contact_effect", moved_a < 1.0,
          f"gravity-free tube with its end 1 mm inside the closed blade: spawned "
          f"{probe_a.round(4).tolist()}, after 60 frames {pa0.round(4).tolist()}, after an "
          f"open/close stroke {pa1.round(4).tolist()} — moved {moved_a:.1f} mm "
          "(the rigid-body tool with collision 'disabled' pushed a tube 54 mm)")
    body_a.set_world_poses(positions=np.array([[0.0, 0.0, -1.0]]))
    step(5)
    tool.open_cutter()
    step(30)
    # Along the hand's Y (the bracket's long side): a tube along X would run
    # back into the Panda's own fingers at the hand's origin and measure THEM.
    probe_c = tool.part_world_positions()["bracket"].copy()
    from simulators.isaac.grasp import _quat_multiply
    along_y = _quat_multiply(np.asarray(hq_), np.array([0.7071068, -0.7071068, 0.0, 0.0]))
    Cylinder(paths="/World/Items/probe_bracket", radii=0.02, heights=0.15, axes="Z",
             positions=np.array([probe_c]), orientations=np.array([along_y]), colors="blue")
    GeomPrim(paths="/World/Items/probe_bracket", apply_collision_apis=True).apply_physics_materials(scene._item_material)
    body_c = RigidPrim(paths="/World/Items/probe_bracket")
    body_c.set_masses(np.array([0.4]))
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Items/probe_bracket")).CreateDisableGravityAttr(True)
    step(60)
    pc = where(body_c)
    moved_c = float(np.linalg.norm(pc - probe_c)) * 1000.0
    check("bracket_has_no_contact_effect", moved_c < 1.0,
          f"gravity-free tube centred on the bracket at {probe_c.round(4).tolist()} -> "
          f"{pc.round(4).tolist()} — moved {moved_c:.1f} mm (rigid-body tool: 49 mm)")
    body_c.set_world_poses(positions=np.array([[0.0, 0.2, -1.0]]))
    step(5)

    # 3. Create two rigid segments while playing.
    created = []
    for i, x in enumerate((0.60, 0.60)):
        path = f"/World/Items/seg_{i}"
        Cylinder(paths=path, radii=0.02, heights=0.15, axes="Z",
                 positions=np.array([[x, -0.05 + 0.1 * i, layout.table_top_z_m + 0.02]]),
                 orientations=np.array([[0.7071068, 0.0, 0.7071068, 0.0]]), colors="green")
        GeomPrim(paths=path, apply_collision_apis=True).apply_physics_materials(scene._item_material)
        body = RigidPrim(paths=path)
        body.set_masses(np.array([0.3]))
        created.append((path, body))
    step(30)
    poses = [where(body).round(3).tolist() for _, body in created]
    check("segments_created_while_playing", True, f"poses {poses}")
    arm_ok("arm_controllable_after_creation")
    try:
        robot.close_gripper(); step(20); robot.open_gripper(); step(20)
        check("gripper_after_creation", True, "gripper commanded")
    except Exception as exc:                                     # noqa: BLE001
        check("gripper_after_creation", False, repr(exc))

    # 3b. THE DISCRETE CUT ON A RESTING TUBE, no arm involved: both segments
    #     must stay where they were spawned. A segment that drifts or spins
    #     here is a spawn-geometry problem (mass, contact, pose), not a grasp
    #     or tool interaction. item-004 is the untouched control.
    try:
        before = scene.item_world_pose("item-003")
        poses = scene.split_item("item-003", cut_offset_m=0.08, kerf_m=0.003,
                                 segment_ids=["item-003-s1", "item-003-s2"],
                                 segment_lengths_m=[0.077, 0.120])
        spawned = {k: np.asarray(v[0], dtype=float) for k, v in poses.items() if k != "parent"}
        drift = {}
        for n in range(120):
            step(1)
            if n in (0, 30, 60, 119):
                for seg in ("item-003-s1", "item-003-s2", "item-004"):
                    p = scene.item_world_pose(seg)
                    v = scene.item_velocities(seg)
                    print(f"TOOL-DIAG split frame {n + 1} {seg}: at {np.round(p[0], 4).tolist()} "
                          f"v {np.round(v[0], 4).tolist()} w {np.round(v[1], 3).tolist()}", flush=True)
        for seg, at in spawned.items():
            now = scene.item_world_pose(seg)[0]
            drift[seg] = round(float(np.linalg.norm(now - at)) * 1000.0, 1)
        check("split_segments_rest_where_spawned", max(drift.values()) < 2.0,
              f"parent was at {np.round(before[0], 4).tolist()}; segment drift after 120 "
              f"frames {drift} mm")
    except Exception as exc:                                     # noqa: BLE001
        check("split_segments_rest_where_spawned", False, f"{type(exc).__name__}: {exc}")
    arm_ok("arm_controllable_after_split")

    # 3b'. A DISMANTLING SPLIT: the segment named as fixed stays kinematic and
    #      immobile while the other one is dragged past it, wakes nothing, and
    #      the released one is an ordinary dynamic body.
    try:
        poses = scene.split_item("item-004", cut_offset_m=0.075, kerf_m=0.003,
                                 segment_ids=["item-004-s1", "item-004-s2"],
                                 segment_lengths_m=[0.072, 0.125],
                                 fixed_segment_id="item-004-s2")
        fixed_at = np.asarray(poses["item-004-s2"][0], dtype=float)
        stage_ = stage_utils.get_current_stage(backend="usd")
        kin = UsdPhysics.RigidBodyAPI(stage_.GetPrimAtPath(item_path("item-004-s2"))
                                      ).GetKinematicEnabledAttr().Get()
        released = scene.items["item-004-s1"]
        p1, q1 = scene.item_world_pose("item-004-s1")
        for n in range(90):
            released.set_world_poses(positions=np.array([[p1[0], p1[1] + 0.0002 * n, p1[2] + 0.0004 * n]]),
                                     orientations=np.array([q1]))
            released.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            step(1)
        now = np.asarray(scene.item_world_pose("item-004-s2")[0], dtype=float)
        v = scene.item_velocities("item-004-s2")
        moved = float(np.linalg.norm(now - fixed_at)) * 1000.0
        check("dismantling_fixed_remainder_stays_installed", bool(kin) and moved < 0.5,
              f"fixed segment kinematic={kin}, moved {moved:.2f} mm while the released segment "
              f"was carried past it (v {np.round(v[0], 4).tolist()})")
        released.set_world_poses(positions=np.array([[0.0, -0.2, -1.0]]))
        step(5)
    except Exception as exc:                                     # noqa: BLE001
        check("dismantling_fixed_remainder_stays_installed", False, f"{type(exc).__name__}: {exc}")

    # 3c. CONTACT OFFSET ACROSS THE KERF. The retained segment is lifted with
    #     only the kerf between its cut face and the remainder's. If PhysX's
    #     contact offset is wider than that gap, the two faces count as
    #     touching and the rising segment drags the remainder through phantom
    #     friction. Measured here by moving one segment kinematically past the
    #     other exactly as the arm does (slowly up, a little sideways), first
    #     with the scene's default offsets, then with a 1 mm contact offset (sum 2 mm, below the 3 mm kerf).
    def drag_test(label):
        s1, s2 = scene.items["item-003-s1"], scene.items["item-003-s2"]
        p1, _ = scene.item_world_pose("item-003-s1")
        p2, q2 = scene.item_world_pose("item-003-s2")
        start = p1.copy()
        for n in range(100):
            s2.set_world_poses(positions=np.array([[p2[0], p2[1] + 0.0001 * n, p2[2] + 0.0003 * n]]),
                               orientations=np.array([q2]))
            s2.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            step(1)
        now, _ = scene.item_world_pose("item-003-s1")
        v = scene.item_velocities("item-003-s1")
        drift = float(np.linalg.norm(now - start)) * 1000.0
        print(f"TOOL-DIAG drag {label}: remainder {start.round(4).tolist()} -> {now.round(4).tolist()} "
              f"({drift:.1f} mm) w {np.round(v[1], 2).tolist()}", flush=True)
        # put both back at rest where they were
        s2.set_world_poses(positions=np.array([p2]), orientations=np.array([q2]))
        s2.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        s1.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        step(30)
        return drift

    try:
        from pxr import PhysxSchema
        offsets = {}
        for seg in ("item-003-s1", "item-003-s2"):
            prim = stage.GetPrimAtPath(item_path(seg))
            attr = prim.GetAttribute("physxCollision:contactOffset")
            offsets[seg] = attr.Get() if attr and attr.HasAuthoredValue() else "default"
        drift_default = drag_test(f"default contact offset {offsets}")
        for seg in ("item-003-s1", "item-003-s2"):
            api = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(item_path(seg)))
            api.CreateContactOffsetAttr(0.001)
            api.CreateRestOffsetAttr(0.0)
        step(5)
        drift_small = drag_test("contact offset 1 mm")
        check("kerf_wider_than_contact_offset_stops_phantom_drag",
              drift_small < 2.0,
              f"remainder drift while the other segment is moved past it: "
              f"{drift_default:.1f} mm with the default offsets, {drift_small:.1f} mm with a 1 mm "
              "contact offset (kerf 3 mm)")
    except Exception as exc:                                     # noqa: BLE001
        check("kerf_wider_than_contact_offset_stops_phantom_drag", False, f"{type(exc).__name__}: {exc}")

    # 4. Deactivate an existing tube while playing.
    victim = item_path("item-001")
    prim = stage.GetPrimAtPath(victim)
    try:
        UsdGeom.Imageable(prim).MakeInvisible()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        body = scene.items["item-001"]
        body.set_world_poses(positions=np.array([[0.0, 0.0, -1.0]]),
                             orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
        body.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        step(30)
        p = where(body)
        check("tube_deactivated_while_playing", p[2] < -0.5, f"parked at {p.round(3).tolist()}")
    except Exception as exc:                                     # noqa: BLE001
        check("tube_deactivated_while_playing", False, repr(exc))
    arm_ok("arm_controllable_after_deactivation")

    # 5. Weld a created body to the hand, then release it.
    try:
        grasp = GraspJoint()
        hp2, hq2 = robot.get_tcp_pose()
        path, body = created[0]
        p, q = body.get_world_poses()
        grasp.attach(hand_path=hand_path, item_path=path, item_id="seg_0",
                     hand_position=hp2, hand_orientation=hq2,
                     item_position=np.asarray(p.numpy()[0]),
                     item_orientation=np.asarray(q.numpy()[0]))
        offset = grasp.offset_in_hand_m.copy()
        step(30)
        grasp.detach()
        step(10)
        check("weld_new_body", np.all(np.isfinite(offset)) and np.all(grasp.offset_in_hand_m == 0),
              f"attached with hand-frame offset {offset.round(4).tolist()} and detached")
    except Exception as exc:                                     # noqa: BLE001
        check("weld_new_body", False, repr(exc))
    arm_ok("arm_controllable_after_weld")

    # 6. LAST: delete a rigid prim while playing (expected to invalidate views).
    try:
        stage.RemovePrim(item_path("item-002"))
        step(10)
        dof = robot.get_joint_state()
        check("delete_while_playing_keeps_arm", np.all(np.isfinite(dof)),
              "arm still readable after RemovePrim")
    except Exception as exc:                                     # noqa: BLE001
        check("delete_while_playing_keeps_arm", False, f"{type(exc).__name__}: {exc}")

    failed = [n for n, ok, _ in results if not ok]
    print(f"TOOL-CHECK-DONE {len(results) - len(failed)}/{len(results)} passed; "
          f"failed: {failed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
