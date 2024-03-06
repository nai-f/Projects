
// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.2;

contract Ticket {
    mapping(address => uint256) public owners;
    mapping (address owner => mapping (address mover => uint amount)) public movers;
    string public name;
    uint public  totalSupply;

    constructor() {
        name = 'Paradise';
        totalSupply = 20000;
        owners[msg.sender] = totalSupply;
    }

    function move (address to, uint amount ) public {
        require(owners[msg.sender] >= amount, "NOt enough tickets");
        owners[msg.sender] -= amount;
        owners[to] += amount;
    }

    function allow ( address mover, uint amount )public {
        movers[msg.sender] [mover] += amount;
    }

    function moveFrom (address from, address to, uint amount) public {
        require(owners[from] >= amount,"not enough tickers");
        require(movers[from][msg.sender] >= amount,"not enough allownace");

        movers[from][msg.sender] -= amount;
        owners[from] -= amount;
        owners[to] += amount;

    }
}
