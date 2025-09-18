import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose


class CommandNode(Node):
    def __init__(self):
        super().__init__('command_node')

        # Publisher: 初始定位
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # Action Client: Navigation2
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 模擬 LLM: 指令 -> 座標
        self.command_to_goal = {
            # "去充電站": (1.0, 0.5),
            # "去廚房": (2.5, -1.2),
            # "去客廳": (0.0, 2.0),
            "1": (-4.3018, 3.6791),
            "2": (6.7558, 2.9908),
            "3": (7.557, -2.0813),
        }

        # 發布一次初始定位
        self.publish_initial_pose(x=-0.41, y=0.95, yaw=1.0)

        self.get_logger().info("指令輸入節點啟動，輸入文字並按 Enter")
        self.run()

    def publish_initial_pose(self, x=0.0, y=0.0, yaw=0.0):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        # 位置
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        # 協方差 (簡單給一個合理值，表示定位不確定性)
        msg.pose.covariance[0] = 0.25  # x 方向
        msg.pose.covariance[7] = 0.25  # y 方向
        msg.pose.covariance[35] = 0.0685  # yaw 方向

        self.initialpose_pub.publish(msg)
        self.get_logger().info(f"已發布初始定位: x={x}, y={y}, yaw={yaw}")

    def run(self):
        while rclpy.ok():
            try:
                command = input("請輸入指令: ")
                if command.strip() == "":
                    continue

                self.get_logger().info(f"收到指令: {command}")

                if command in self.command_to_goal:
                    x, y = self.command_to_goal[command]
                    self.send_goal(x, y)
                else:
                    self.get_logger().warn("無法解析指令，請再試一次")

            except KeyboardInterrupt:
                self.get_logger().info("結束輸入")
                break

    def send_goal(self, x, y):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0

        self._action_client.wait_for_server()

        self.get_logger().info(f"發送導航目標: x={x}, y={y}")
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('導航目標被拒絕！')
            return

        self.get_logger().info('導航目標已接受')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        if status == 4:  # SUCCEEDED
            self.get_logger().info('導航成功')
        else:
            self.get_logger().warn(f'導航失敗，狀態碼: {status}')


def main(args=None):
    rclpy.init(args=args)
    node = CommandNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()