from threading import Thread
from socketIO_client_nexus import SocketIO

host = 'https://damp-harbor-42374.herokuapp.com'
port = 443


class TwoWayClient(object):
    def __init__(self):
        self.socketIO = SocketIO(host, port)
        
        def on_connect():
            print("Server connected")

        def on_disconnect(*args):
            print("Server disconnected")

        def on_reconnect():
            print("Server reconnected")

        def on_time(*args):
            print(args[0])

        self.socketIO.on('connect', on_connect) # This is not working
        self.socketIO.on('disconnect', on_disconnect) # This works
        self.socketIO.on('reconnect', on_reconnect) # This is not working
        self.socketIO.on('time', on_time)

        self.receive_events_thread = Thread(target=self._receive_events_thread)
        self.receive_events_thread.daemon = True
        self.receive_events_thread.start()

        while True:
            some_input = input("Please input: ")
            self.socketIO.emit('custom', some_input)

    def _receive_events_thread(self):
        self.socketIO.wait()


def main():
    TwoWayClient()


if __name__ == "__main__":
    main()
