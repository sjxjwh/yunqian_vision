// 最小消费者：连上仿真共享内存，读一帧，按无解回写云台指令。
// 用来验证 io/sim 与仿真侧布局的对接，不依赖推理。
#include <iostream>

#include "io/sim/talos_shm.hpp"

int main()
{
  io::talos::ShmClient client;
  client.connect();

  io::talos::Frame frame;
  if (!client.recv(frame)) {
    std::cerr << "FAIL: no frame\n";
    return 1;
  }

  auto bgr = frame.bgr.at<cv::Vec3b>(0, 0);
  std::cout << "seq=" << frame.seq << " stamp=" << frame.timestamp_ns << " bgr0=" << (int)bgr[0]
            << "," << (int)bgr[1] << "," << (int)bgr[2] << " fx=" << frame.camera_matrix(0, 0)
            << " qw=" << frame.gimbal_quat.w() << "\n";

  bool ok = frame.seq == 7 && bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 255 &&
            std::abs(frame.camera_matrix(0, 0) - 1304.673) < 1e-3;

  io::talos::GimbalCmd cmd{};
  cmd.timestamp_ns = frame.timestamp_ns;
  cmd.yaw_deg = 0;
  cmd.pitch_deg = -90;
  cmd.distance_m = io::talos::NO_SOLUTION_DISTANCE;
  client.send(cmd);

  return ok ? 0 : 1;
}
