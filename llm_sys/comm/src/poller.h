//
// Created by meiyixuan on 2024/4/15.
//

#ifndef ZMQ_COMM_POLLER_H
#define ZMQ_COMM_POLLER_H

#include <utility>
#include <zmq.hpp>
#include <iostream>
#include <vector>

#include "utils.h"
#include "msg.h"


class PollingClient {
public:
    PollingClient(zmq::context_t &_context, std::vector<std::string> &_server_addresses) : context(_context) {
        // initialize socket
        server_addresses = _server_addresses;
        subscriber = zmq::socket_t(context, ZMQ_SUB);
        for (const auto &addr: server_addresses) {
            subscriber.connect(addr);
            std::cout << "Receiver connected to address: " << addr << "\n";
        }
        subscriber.set(zmq::sockopt::subscribe, "");  // subscribe all messages
    }

    Header poll_once(zmq::message_t &buffer_msg, int time_out = 10) {
        // header will be returned
        // buffer will be put into buffer_msg, access it with buffer_msg.data()

        zmq::pollitem_t items[] = {
                {subscriber, 0, ZMQ_POLLIN, 0},
        };
        zmq::poll(items, 1, std::chrono::milliseconds(time_out));
        if ((items[0].revents & ZMQ_POLLIN) != 0) {
            zmq::message_t header_msg;
            auto _val_1 = subscriber.recv(header_msg, zmq::recv_flags::none);
            auto _val_2 = subscriber.recv(buffer_msg, zmq::recv_flags::none);
            Header header = Header::deserialize(header_msg);
            return header;
        }

        // return an empty header, indicating no message received
        return {};
    }

private:
    std::vector<std::string> server_addresses;
    zmq::context_t &context;
    zmq::socket_t subscriber;
};


class PollServer {
public:
    PollServer(zmq::context_t &_context, std::string &bind_address) : context(_context) {
        // initialize socket
        publisher = zmq::socket_t(context, ZMQ_PUB);
        publisher.bind(bind_address);
        std::cout << "Sender bound to address: " << bind_address << "\n";
    }

    void send(const Header &header, zmq::message_t &buffer_msg) {
        // header: header of the message
        // buffer_msg: message content
        zmq::message_t header_msg = header.serialize();
        publisher.send(header_msg, zmq::send_flags::sndmore);
        publisher.send(buffer_msg, zmq::send_flags::none);
    }

private:
    zmq::context_t &context;
    zmq::socket_t publisher;
};


#endif //ZMQ_COMM_POLLER_H
