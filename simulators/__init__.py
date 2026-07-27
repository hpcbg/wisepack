"""Execution-backend simulators for WISEPACK.

One subpackage per physical execution backend. Today that is ``isaac`` (NVIDIA
Isaac Sim 6.0.1); a real robot cell would sit here as a sibling, answering the
same two ROS 2 topics defined in ``wisepack_core.isaac_contract``.

THIS IS A PACKAGE FOR A REASON, and the reason is a measured failure. The
modules were originally top-level scripts imported by bare name — ``config``,
``bridge``, ``result``. Isaac Sim's bundled interpreter carries a large
pre-bundled site-packages tree, and one of them owns that name:

    from config import LOG_APP, AppConfig, ...
      File ".../omni.pip.compute/pip_prebundle/cv2/config.py", line 4
    NameError: name 'LOADER_DIR' is not defined

OpenCV's ``cv2/config.py`` shadowed ours and failed on import. Generic top-level
module names are unsafe inside a host application's interpreter, so everything
here is reached through this package instead.
"""
