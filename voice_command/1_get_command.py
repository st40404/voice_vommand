import rclpy
from rclpy.node import Node

class CommandNode(Node):
    def __init__(self):
        super().__init__('command_node')
        self.get_logger().info("指令輸入節點啟動，輸入文字並按 Enter")
        self.run()

    def run(self):
        while rclpy.ok():
            try:
                command = input("請輸入指令: ")
                if command.strip() == "":
                    continue
                self.get_logger().info(f"收到指令: {command}")
                # 之後在這裡加上: 轉換成座標 → 發送到 Navigation2
            except KeyboardInterrupt:
                self.get_logger().info("結束輸入")
                break

def main(args=None):
    rclpy.init(args=args)
    node = CommandNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
