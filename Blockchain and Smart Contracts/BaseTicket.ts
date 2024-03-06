// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract BaseTicket {
    mapping(address => uint256) public owners;
    mapping(address owner => mapping(address mover => uint amount))
        public movers;
    string public name;
    uint public totalSupply;

    constructor(string memory _name, uint256 _totalSupply) {
        name = _name;
        totalSupply = _totalSupply;
        owners[msg.sender] = totalSupply;
    }

    // called by owners: owner --> to
    function move(address to, uint256 amount) public {
        require(owners[msg.sender] >= amount, "Not enough tickets!");
        owners[msg.sender] -= amount;
        owners[to] += amount;
    }

    // called by owners
    function allow(address mover, uint amount) public {
        movers[msg.sender][mover] += amount;
    }

    // called by movers
    function moveFrom(address from, address to, uint amount) public {
        require(owners[from] >= amount, "Not enough tickets!");
        require(movers[from][msg.sender] >= amount, "Not enough allowance");
        movers[from][msg.sender] -= amount;
        owners[from] -= amount;
        owners[to] += amount;
    }
}
