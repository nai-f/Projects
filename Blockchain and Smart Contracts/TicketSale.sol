// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.2;

import {Ticket} from 'contracts/Ticket.sol';

contract TicketSale{
    address ticket;
    address ticketCreator; //alice address the ticket creator

    constructor(address _ticket){
        ticket = _ticket;
        ticketCreator = msg.sender; // the deployer

    }
    // called by the investors
    function fund(uint amount) public  payable {
        require(msg.value == amount * 1 gwei, "Payment is not correct");
        Ticket(ticket).moveFrom(ticketCreator, msg.sender, amount);
    }
}
