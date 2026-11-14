{ pkgs }:
let
  tools = with pkgs; [
    treefmt
    clang-tools
    ruff
    shfmt
    nixfmt
    taplo
    yamlfmt
    mbake
    python3
    git
  ];
in
{
  inherit tools;
  package = pkgs.writeShellApplication {
    name = "project-format";
    runtimeInputs = tools;
    text = ''
      exec treefmt "$@"
    '';
  };
  check =
    src:
    pkgs.runCommand "formatting-check" { nativeBuildInputs = tools; } ''
      cp -r ${src} source
      chmod -R u+w source
      cd source
      treefmt --no-cache --fail-on-change
      touch "$out"
    '';
}
