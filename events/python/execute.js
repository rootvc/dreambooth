import { defineFunction } from "../tools";

const PYODIDE_VERSION = "0.22.1";
const PYODIDE_URL = `https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/pyodide-${PYODIDE_VERSION}.tar.bz2`;

export default defineFunction(
  "Execute Python Module",
  "dreambooth/python.execute",
  async ({ event }) => {
    //
  }
);
