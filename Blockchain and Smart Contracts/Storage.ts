import { expect } from "chai";
import { ethers } from "hardhat";

describe("Storage", function () {
  beforeEach(async function () {
    const StorageContract = await ethers.getContractFactory("Storage");
    this.Storage = await StorageContract.deploy();
  });

  it("Should retrieve 0", async function () {
    expect(await this.Storage.retrieve()).to.equal(0);
  });

  // I added this to double check

  it("Should retrieve 10", async function () {
    await this.Storage.store(10);
    expect(await this.Storage.retrieve()).to.equal(10);
  });

  it("Should retrieve 15", async function () {
    await this.Storage.store(15);
    expect(await this.Storage.retrieve()).to.equal(15);
  });
});
