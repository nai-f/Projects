import { ethers } from "hardhat";
import { expect } from "chai";
import { loadFixture } from "@nomicfoundation/hardhat-toolbox/network-helpers";

describe("Ticket", function () {
  async function initializeTicket() {
    const [alice, bob, charlie] = await ethers.getSigners();
    const ticketContract = await ethers.deployContract("Ticket");
    return { alice, bob, charlie, ticketContract };
  }

  it("Ticket total supply should be 20000", async function () {
    const { alice, bob, charlie, ticketContract } = await loadFixture(
      initializeTicket
    );
    const totalSupply = await ticketContract.totalSupply();
    expect(totalSupply).to.equal(20000);
  });

  it("Bob should have 10000", async function () {
    const { alice, bob, charlie, ticketContract } = await loadFixture(
      initializeTicket
    );
    await ticketContract.move(bob.address, 10000);
    const amount = await ticketContract.owners(bob.address);
    expect(amount).to.equal(10000);
  });

  it("Should revert with some message", async function () {
    const { alice, bob, charlie, ticketContract } = await loadFixture(
      initializeTicket
    );
    await expect(
      ticketContract.connect(bob).move(charlie.address, 200)
    ).to.be.revertedWith("Not enough tickets!");
  });
});
