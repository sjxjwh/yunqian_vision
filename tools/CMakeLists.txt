cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT 
    exiter.cpp
    extended_kalman_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    plotter.cpp
    web_viewer.cpp
    sim_metrics.cpp
    gimbal_csv_logger.cpp
    trajectory.cpp
    recorder.cpp
    sync_recorder.cpp
    logger.cpp
    pid.cpp
    crc.cpp
)