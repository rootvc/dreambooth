import got from "got";
import { spawn } from "node:child_process";
import fs from "node:fs";
import { pipeline } from "node:stream/promises";
import { defineFunction } from "../tools";

const PYODIDE_VERSION = "0.22.1";
const PYODIDE_URL = `https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/pyodide-${PYODIDE_VERSION}.tar.bz2`;

export default defineFunction(
  "Execute Python Module",
  "dreambooth/python.execute",
  async ({ event }) => {
    await pipeline(
      got.stream.get(PYODIDE_URL),
      fs.createWriteStream("pyodide.tar.bz2")
    );
    await spawn("tar", ["-xjvf", "pyodide.tar.bz2", "-C", "pyodide"]);
  }
);
