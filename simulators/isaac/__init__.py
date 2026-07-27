"""The NVIDIA Isaac Sim 6.0.1 physical execution backend for WISEPACK.

DELIBERATELY EMPTY OF IMPORTS. Two of the modules here — ``config`` and
``result`` — carry no Isaac dependency at all and are unit-tested by the ordinary
test suite on a machine with no GPU and no simulator. Importing the scene, robot
or bridge modules from this file would drag ``isaacsim`` into that import and
make the normal suite require Isaac Sim, which is precisely what the
non-Isaac-CI requirement forbids.

    config    tunables, validated                  — no Isaac
    result    settling and containment verdicts    — no Isaac
    bridge    ROS 2 transport                      — rclpy only
    scene     procedural table / bin / cylinders   — Isaac
    grasp     the temporary fixed joint            — Isaac
    robot     the Panda state machine              — Isaac
    wisepack_isaac  assembly and the main loop     — Isaac (entry point)

See README.md in this directory for the limitations of this iteration, which are
stated first there rather than buried.
"""
