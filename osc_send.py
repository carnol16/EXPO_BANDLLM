from pythonosc import udp_client

class OSC_Sender():
    def __init__(self, address="192.168.1.163"):
        self.address = address
        self.max_port = 9020
        self.td_port = 9030
        self.andrew_port = 9040
        self.max_client = udp_client.SimpleUDPClient(address,9020)
        self.td_client = udp_client.SimpleUDPClient(address,9030)
        self.andrew_client = udp_client.SimpleUDPClient(address,9040)
    def send_message(self, tag, msg):
        if tag[0] != "/":
            print("all tags need to start with a / ")
            return
        if type(msg) != str:
            print("msg must be in string format")
            return
        self.max_client.send_message(tag, msg)
        self.td_client.send_message(tag, msg) 
        self.andrew_client.send_message(tag, msg) 
