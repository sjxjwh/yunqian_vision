#ifndef TOOLS__EXITER_HPP
#define TOOLS__EXITER_HPP

namespace tools
{
class Exiter
{
public:
  Exiter();

  bool exit() const;

  // 不创建信号处理器，只读退出标志。给不能再实例化 Exiter 的等待循环用。
  static bool requested();
};

}  // namespace tools

#endif  // TOOLS__EXITER_HPP