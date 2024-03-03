import socket

def start_client(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    print(f"Connected to server at {host}:{port}")

    while True:
        message = input("Enter your message: ")
        client_socket.sendall(message.encode('utf-8'))

        if message.lower() == 'exit':
            break

    client_socket.close()

if __name__ == "__main__":
    HOST = '127.0.0.1'  # Use your server IP address here
    PORT = 12345       # Use the same port number as in the server

    start_client(HOST, PORT)
