#include "web_viewer.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "tools/logger.hpp"

namespace tools
{
namespace
{

constexpr int MAX_CLIENTS = 4;

std::string http_status(int code, const std::string & text)
{
  return "HTTP/1.1 " + std::to_string(code) + " " + text + "\r\nContent-Length: 0\r\n\r\n";
}

std::string content_type(const std::string & path)
{
  if (path.size() >= 5 && path.substr(path.size() - 5) == ".html") return "text/html; charset=utf-8";
  if (path.size() >= 3 && path.substr(path.size() - 3) == ".js") return "text/javascript";
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") return "model/gltf-binary";
  return "application/octet-stream";
}

// 只允许 web 根目录内的相对路径，拒绝 .. 穿越。
bool safe_path(const std::string & path)
{
  if (path.empty() || path[0] == '/') return false;
  return path.find("..") == std::string::npos;
}

bool send_all(int fd, const char * data, size_t size)
{
  size_t sent = 0;
  while (sent < size) {
    auto n = ::send(fd, data + sent, size - sent, MSG_NOSIGNAL);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool send_all(int fd, const std::string & data) { return send_all(fd, data.data(), data.size()); }

// WebSocket 帧：服务端只发，不收客户端数据，所以不处理掩码。
std::string ws_frame(uint8_t opcode, const std::string & payload)
{
  std::string frame;
  frame.push_back(static_cast<char>(0x80 | opcode));
  auto n = payload.size();
  if (n < 126) {
    frame.push_back(static_cast<char>(n));
  } else if (n < 65536) {
    frame.push_back(126);
    frame.push_back(static_cast<char>((n >> 8) & 0xff));
    frame.push_back(static_cast<char>(n & 0xff));
  } else {
    frame.push_back(127);
    for (int i = 7; i >= 0; i--) frame.push_back(static_cast<char>((n >> (8 * i)) & 0xff));
  }
  frame += payload;
  return frame;
}

std::string ws_accept_key(const std::string & key)
{
  // SHA1("key" + magic) 的 base64。手写避免引入加密库。
  auto input = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
  uint64_t bits = input.size() * 8;
  input.push_back(static_cast<char>(0x80));
  while (input.size() % 64 != 56) input.push_back(0);
  for (int i = 7; i >= 0; i--) input.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));

  for (size_t chunk = 0; chunk < input.size(); chunk += 64) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
      w[i] = (static_cast<uint8_t>(input[chunk + i * 4]) << 24) |
             (static_cast<uint8_t>(input[chunk + i * 4 + 1]) << 16) |
             (static_cast<uint8_t>(input[chunk + i * 4 + 2]) << 8) |
             static_cast<uint8_t>(input[chunk + i * 4 + 3]);
    }
    for (int i = 16; i < 80; i++) {
      auto x = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
      w[i] = (x << 1) | (x >> 31);
    }
    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    for (int i = 0; i < 80; i++) {
      uint32_t f, k;
      if (i < 20) {
        f = (b & c) | (~b & d);
        k = 0x5A827999;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDC;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6;
      }
      auto temp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
      e = d;
      d = c;
      c = (b << 30) | (b >> 2);
      b = a;
      a = temp;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
  }

  uint8_t digest[20];
  uint32_t hs[5] = {h0, h1, h2, h3, h4};
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++) digest[i * 4 + j] = (hs[i] >> (24 - 8 * j)) & 0xff;

  static const char table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (int i = 0; i < 20; i += 3) {
    uint32_t v = digest[i] << 16;
    if (i + 1 < 20) v |= digest[i + 1] << 8;
    if (i + 2 < 20) v |= digest[i + 2];
    out.push_back(table[(v >> 18) & 63]);
    out.push_back(table[(v >> 12) & 63]);
    out.push_back(i + 1 < 20 ? table[(v >> 6) & 63] : '=');
    out.push_back(i + 2 < 20 ? table[v & 63] : '=');
  }
  return out;
}

}  // namespace

struct WebViewer::Impl
{
  int listen_fd = -1;
  std::string web_root;
  std::thread thread;
  std::atomic<bool> stop{false};

  std::mutex mutex;
  std::deque<std::pair<uint64_t, std::string>> state_queue;
  cv::Mat latest_bgr;
  uint64_t generation = 0;
  uint64_t image_generation = 0;

  std::mutex jpeg_mutex;
  std::string latest_jpeg_frame;
  uint64_t jpeg_generation = 0;

  struct Client
  {
    int fd = -1;
    uint64_t seen = 0;
    uint64_t seen_jpeg = 0;
  };
  std::mutex clients_mutex;
  std::vector<Client> clients;
  std::atomic<int> client_count{0};

  std::mutex api_mutex;
  ApiHandler api_handler;

  void accept_loop();
  void handle(int fd);
  void serve_file(int fd, const std::string & path);
  void jpeg_loop();
  void push_loop();
};

WebViewer::WebViewer(uint16_t port, std::string web_root) : impl_(new Impl)
{
  impl_->web_root = std::move(web_root);

  impl_->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  int reuse = 1;
  ::setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (::bind(impl_->listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
      ::listen(impl_->listen_fd, 4) < 0) {
    tools::logger()->error("[Viewer] 端口 {} 绑定失败", port);
    ::close(impl_->listen_fd);
    impl_->listen_fd = -1;
    return;
  }

  impl_->thread = std::thread([this] { impl_->accept_loop(); });
  tools::logger()->info("[Viewer] http://0.0.0.0:{}", port);
}

WebViewer::~WebViewer()
{
  impl_->stop = true;
  if (impl_->listen_fd >= 0) ::shutdown(impl_->listen_fd, SHUT_RDWR);
  if (impl_->thread.joinable()) impl_->thread.join();
  if (impl_->listen_fd >= 0) ::close(impl_->listen_fd);
  std::lock_guard<std::mutex> lock(impl_->clients_mutex);
  for (auto & c : impl_->clients)
    if (c.fd >= 0) ::close(c.fd);
  delete impl_;
}

void WebViewer::set_api_handler(ApiHandler handler)
{
  std::lock_guard<std::mutex> lock(impl_->api_mutex);
  impl_->api_handler = std::move(handler);
}

bool WebViewer::has_clients() const
{
  return impl_->client_count.load(std::memory_order_relaxed) > 0;
}

void WebViewer::publish(const cv::Mat & bgr, const ViewerFrame & frame)
{
  if (impl_->client_count.load(std::memory_order_relaxed) == 0) return;

  nlohmann::json armors = nlohmann::json::array();
  for (auto & a : frame.armors) armors.push_back({a[0], a[1], a[2], a[3], a[4], a[5]});

  nlohmann::json armor_pixels = nlohmann::json::array();
  for (auto & p : frame.armor_pixels) {
    armor_pixels.push_back({p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]});
  }

  nlohmann::json detections = nlohmann::json::array();
  for (auto & d : frame.detections) {
    detections.push_back(
      {d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]});
  }

  nlohmann::json bullets = nlohmann::json::array();
  for (const auto & b : frame.bullets) {
    bullets.push_back({b[0], b[1], b[2]});
  }

  nlohmann::json hit_events = nlohmann::json::array();
  for (const auto & h : frame.hit_events) {
    hit_events.push_back(
      {{"id", h.id}, {"x", h.x}, {"y", h.y}, {"z", h.z}, {"armor_id", h.armor_id}});
  }

  nlohmann::json corrector_val = nullptr;
  if (!frame.corrector_json.empty()) {
    try {
      corrector_val = nlohmann::json::parse(frame.corrector_json);
    } catch (const std::exception &) {
      corrector_val = nullptr;
    }
  }

  nlohmann::json state = {
    {"stamp_ns", frame.stamp_ns},
    {"gimbal", {frame.gimbal_yaw, frame.gimbal_pitch}},
    {"muzzle", {frame.muzzle_yaw, frame.muzzle_pitch}},
    {"self_pos", {frame.self_pos[0], frame.self_pos[1], frame.self_pos[2]}},
    {"geom",
     {{"camera", {frame.camera_in_gimbal[0], frame.camera_in_gimbal[1], frame.camera_in_gimbal[2]}},
      {"muzzle", {frame.muzzle_in_gimbal[0], frame.muzzle_in_gimbal[1], frame.muzzle_in_gimbal[2]}}}},
    {"tracked", frame.tracked},
    {"target", frame.target_name},
    {"target_big", frame.target_is_big},
    {"target_center", {frame.target_center[0], frame.target_center[1], frame.target_center[2]}},
    {"target_vel", {frame.target_vel[0], frame.target_vel[1], frame.target_vel[2]}},
    {"target_w", frame.target_w},
    {"armors", armors},
    {"armor_pixels", armor_pixels},
    {"current_armor", frame.current_armor},
    {"detections", detections},
    {"aim", {frame.aim_valid, frame.aim_x, frame.aim_y, frame.aim_z}},
    {"command", {frame.control, frame.command_yaw, frame.command_pitch, frame.shoot}},
    {"ref", {frame.ref_yaw, frame.ref_pitch}},
    {"metrics",
     {{"shots", frame.shots},
      {"hits", frame.hits},
      {"resolved", frame.resolved_shots},
      {"hit_rate", frame.hit_rate},
      {"fire_duty", frame.fire_duty},
      {"yaw_err_rms", frame.yaw_err_rms},
      {"has_sim", frame.has_sim_metrics},
      {"sim_shots", frame.sim_shots},
      {"sim_hits", frame.sim_hits},
      {"sim_resolved", frame.sim_resolved_shots},
      {"sim_hit_rate", frame.sim_hit_rate}}},
    {"bullets", bullets},
    {"hit_events", hit_events},
    {"corrector", corrector_val},
  };

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!bgr.empty()) {
    impl_->latest_bgr = bgr.clone();
    impl_->image_generation++;
  }
  state["img_id"] = static_cast<uint32_t>(impl_->image_generation);
  impl_->generation++;
  impl_->state_queue.emplace_back(impl_->generation, state.dump());
  if (impl_->state_queue.size() > 16) impl_->state_queue.pop_front();
}

void WebViewer::Impl::accept_loop()
{
  std::thread jpeg_encoder([this] { jpeg_loop(); });
  std::thread pusher([this] { push_loop(); });
  while (!stop) {
    auto fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) continue;
    std::thread(&Impl::handle, this, fd).detach();
  }
  pusher.join();
  jpeg_encoder.join();
}

void WebViewer::Impl::serve_file(int fd, const std::string & path)
{
  if (!safe_path(path)) {
    send_all(fd, http_status(403, "Forbidden"));
    return;
  }
  auto full = web_root + "/" + path;
  FILE * file = std::fopen(full.c_str(), "rb");
  if (!file) {
    send_all(fd, http_status(404, "Not Found"));
    return;
  }
  std::fseek(file, 0, SEEK_END);
  auto size = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  std::string body(size, '\0');
  std::fread(body.data(), 1, size, file);
  std::fclose(file);

  std::ostringstream header;
  header << "HTTP/1.1 200 OK\r\nContent-Type: " << content_type(path)
         << "\r\nContent-Length: " << body.size() << "\r\nCache-Control: no-cache\r\n\r\n";
  send_all(fd, header.str());
  send_all(fd, body);
}

void WebViewer::Impl::handle(int fd)
{
  char buf[4096];
  auto n = ::recv(fd, buf, sizeof(buf) - 1, 0);
  if (n <= 0) {
    ::close(fd);
    return;
  }
  buf[n] = '\0';
  std::string request(buf);

  auto line_end = request.find("\r\n");
  auto first = request.substr(0, line_end);
  std::string path = "index.html";
  auto sp1 = first.find(' ');
  auto sp2 = first.find(' ', sp1 + 1);
  if (sp1 != std::string::npos && sp2 != std::string::npos) {
    auto p = first.substr(sp1 + 1, sp2 - sp1 - 1);
    if (p != "/") path = p.substr(1);
  }

  if (path == "ws") {
    auto key_pos = request.find("Sec-WebSocket-Key: ");
    if (key_pos == std::string::npos) {
      ::close(fd);
      return;
    }
    key_pos += 19;
    auto key = request.substr(key_pos, request.find("\r\n", key_pos) - key_pos);
    std::ostringstream handshake;
    handshake << "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: "
                 "Upgrade\r\nSec-WebSocket-Accept: "
              << ws_accept_key(key) << "\r\n\r\n";
    if (!send_all(fd, handshake.str())) {
      ::close(fd);
      return;
    }
    std::lock_guard<std::mutex> lock(clients_mutex);
    if (static_cast<int>(clients.size()) >= MAX_CLIENTS) {
      ::close(fd);
      return;
    }
    clients.push_back({fd, 0});
    client_count.store(static_cast<int>(clients.size()), std::memory_order_relaxed);
    return;
  }

  if (path.rfind("api/", 0) == 0) {
    ApiHandler fn;
    {
      std::lock_guard<std::mutex> lock(api_mutex);
      fn = api_handler;
    }
    if (!fn) {
      send_all(fd, http_status(404, "Not Found"));
      ::close(fd);
      return;
    }
    std::string body = fn(path);
    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: "
           << body.size() << "\r\nCache-Control: no-cache\r\n\r\n";
    send_all(fd, header.str());
    send_all(fd, body);
    ::close(fd);
    return;
  }

  serve_file(fd, path);
  ::close(fd);
}

void WebViewer::Impl::jpeg_loop()
{
  uint64_t last_encoded_img_gen = 0;
  while (!stop) {
    std::this_thread::sleep_for(std::chrono::milliseconds(33));
    if (client_count.load(std::memory_order_relaxed) == 0) continue;

    cv::Mat bgr_to_encode;
    uint32_t encoded_img_id = 0;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!latest_bgr.empty() && image_generation != last_encoded_img_gen) {
        bgr_to_encode = latest_bgr;
        last_encoded_img_gen = image_generation;
        encoded_img_id = static_cast<uint32_t>(image_generation);
      }
    }
    if (bgr_to_encode.empty()) continue;

    std::vector<uint8_t> jpeg;
    cv::imencode(".jpg", bgr_to_encode, jpeg, {cv::IMWRITE_JPEG_QUALITY, 60});
    if (jpeg.empty()) continue;

    std::string payload;
    payload.resize(4 + jpeg.size());
    std::memcpy(payload.data(), &encoded_img_id, 4);
    std::memcpy(payload.data() + 4, jpeg.data(), jpeg.size());
    auto frame = ws_frame(0x2, payload);
    std::lock_guard<std::mutex> lock(jpeg_mutex);
    latest_jpeg_frame = std::move(frame);
    jpeg_generation++;
  }
}

void WebViewer::Impl::push_loop()
{
  auto next_tick = std::chrono::steady_clock::now();
  while (!stop) {
    next_tick += std::chrono::milliseconds(10);
    std::this_thread::sleep_until(next_tick);
    if (client_count.load(std::memory_order_relaxed) == 0) continue;

    std::vector<std::pair<uint64_t, std::string>> states;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!state_queue.empty()) {
        states.assign(state_queue.begin(), state_queue.end());
        state_queue.clear();
      }
    }

    std::string binary_frame;
    uint64_t cur_jpeg_gen = 0;
    {
      std::lock_guard<std::mutex> lock(jpeg_mutex);
      if (!latest_jpeg_frame.empty()) {
        binary_frame = latest_jpeg_frame;
        cur_jpeg_gen = jpeg_generation;
      }
    }

    if (states.empty() && binary_frame.empty()) continue;

    std::vector<std::pair<uint64_t, std::string>> text_frames;
    text_frames.reserve(states.size());
    for (const auto & s : states) {
      text_frames.emplace_back(s.first, ws_frame(0x1, s.second));
    }

    std::lock_guard<std::mutex> lock(clients_mutex);
    for (auto it = clients.begin(); it != clients.end();) {
      bool ok = true;
      for (const auto & tf : text_frames) {
        if (tf.first > it->seen) {
          ok = send_all(it->fd, tf.second);
          if (!ok) break;
          it->seen = tf.first;
        }
      }
      if (ok && !binary_frame.empty() && cur_jpeg_gen != it->seen_jpeg) {
        ok = send_all(it->fd, binary_frame);
        if (ok) it->seen_jpeg = cur_jpeg_gen;
      }
      if (!ok) {
        ::close(it->fd);
        it = clients.erase(it);
        continue;
      }
      ++it;
    }
    client_count.store(static_cast<int>(clients.size()), std::memory_order_relaxed);
  }
}

}  // namespace tools
