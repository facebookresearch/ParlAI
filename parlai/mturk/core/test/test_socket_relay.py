from threading import Thread
from socketIO_client_nexus import SocketIO

host = 'https://frozen-tundra-27407.herokuapp.com'
port = 443


class TwoWayClient(object):
    def __init__(self):
        self.socketIO = SocketIO(host, port)
        
        def on_socket_open(*args):
            print("on_socket_open: ", args[0])
            self.socketIO.emit('agent_alive', {'task_group_id': 'test_group', 'agent_id': '[World]'})

        def on_disconnect(*args):
            print("Server disconnected", args[0])

        def on_new_messgae(*args):
            print(args[0])

        self.socketIO.on('socket_open', on_socket_open)
        self.socketIO.on('disconnect', on_disconnect) # This works
        self.socketIO.on('new_message', on_new_messgae)

        self.receive_events_thread = Thread(target=self._receive_events_thread)
        self.receive_events_thread.daemon = True
        self.receive_events_thread.start()

        while True:
            some_input = input("Hit ENTER to send message:")
            self.socketIO.emit('new_message', {'task_group_id': 'test_group', 'receiver_agent_id': 'Worker'})

    def _receive_events_thread(self):
        self.socketIO.wait()


def main():
    TwoWayClient()


if __name__ == "__main__":
    main()
