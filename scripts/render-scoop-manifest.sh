#!/usr/bin/env bash
# Print the Scoop manifest (vex.json) for one release. The release workflow
# commits it to tenatarika/scoop-bucket; CI installs from it to test it.
# Usage: scripts/render-scoop-manifest.sh <version> <sha256 of vex archive> <sha256 of vex-mcp archive>
set -euo pipefail
version="$1" sha_vex="$2" sha_mcp="$3"
base="https://github.com/tenatarika/vex/releases/download"

jq -n --arg v "$version" --arg sv "$sha_vex" --arg sm "$sha_mcp" --arg base "$base" '{
  version: $v,
  description: "Fast hybrid structural + semantic code search for humans and AI agents, with an MCP server",
  homepage: "https://github.com/tenatarika/vex",
  license: "MIT",
  notes: "vex-mcp is the MCP server; point your agent at vex-mcp.exe. Update with: scoop update vex (not vex self-update).",
  architecture: {
    "64bit": {
      url: [
        "\($base)/v\($v)/vex-x86_64-pc-windows-msvc.tar.gz",
        "\($base)/v\($v)/vex-mcp-x86_64-pc-windows-msvc.tar.gz"
      ],
      hash: [$sv, $sm]
    }
  },
  bin: ["vex.exe", "vex-mcp.exe"],
  checkver: "github",
  autoupdate: {
    architecture: {
      "64bit": {
        url: [
          "\($base)/v$version/vex-x86_64-pc-windows-msvc.tar.gz",
          "\($base)/v$version/vex-mcp-x86_64-pc-windows-msvc.tar.gz"
        ]
      }
    }
  }
}'
