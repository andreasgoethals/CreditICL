#!/bin/bash
# =============================================================================
#  Auto-activate this project's venv whenever your shell is inside the repo.
#
#  INSTALL IT ONCE:
#
#      bash scripts/slurm/shell_hook.sh --install
#
#  That appends ONE line to ~/.bashrc which sources this file. The hook itself
#  stays here, in the repo, so `git pull` keeps it up to date and ~/.bashrc never
#  has to be edited again.
#
#  Uninstall:  bash scripts/slurm/shell_hook.sh --uninstall
#  Check:      bash scripts/slurm/shell_hook.sh --status
#
#  This is for INTERACTIVE shells only. Slurm jobs do not read ~/.bashrc
#  reliably, which is why scripts/slurm/_activate_env.sh activates the venv
#  itself and does not depend on this.
# =============================================================================

# --- the hook ---------------------------------------------------------------
# Defined whether this file is sourced or run, so `--status` can inspect it.

_crediticl_auto_venv() {
    local repo="${VSC_DATA}/CreditICL"
    local venv="${repo}/.venv-${VSC_ARCH_LOCAL:-generic}"

    # Fall back to any venv in the repo. Login nodes differ in $VSC_ARCH_LOCAL, and a venv
    # built on a compatible architecture beats no environment at all.
    if [[ ! -x "$venv/bin/python" ]]; then
        local found
        for found in "$repo"/.venv-* "$repo/.venv"; do
            [[ -x "$found/bin/python" ]] && { venv="$found"; break; }
        done
    fi

    case "$PWD/" in
        "$repo"/*)
            if [[ -x "$venv/bin/python" && "${VIRTUAL_ENV:-}" != "$venv" ]]; then
                # Another project's venv may already be active (TabPFNCredit's was, and it
                # kept winning the PATH race). Stand it down before taking over.
                if [[ -n "${VIRTUAL_ENV:-}" ]] && declare -f deactivate >/dev/null 2>&1; then
                    deactivate 2>/dev/null
                fi
                # The module that BUILT this venv, recorded by setup_venv.sh. Read from disk
                # rather than hard-coded: module trees on VSC are per-architecture, so a pinned
                # name resolves on one login node and not another.
                local mod
                mod="$(cat "$venv/.python_module" 2>/dev/null)"
                [[ -n "$mod" ]] && module load "$mod" 2>/dev/null
                # shellcheck disable=SC1091
                source "$venv/bin/activate"
            fi
            ;;
        *)
            # Only stand down OUR venv. Someone else's is left alone.
            if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" == "$repo"/* ]]; then
                deactivate 2>/dev/null
            fi
            ;;
    esac
}

# Register it. PROMPT_COMMAND, not an overridden `cd`: it fires before every prompt, so it also
# works after `pushd`, inside a subshell, and when you arrive via a symlink.
case "${PROMPT_COMMAND:-}" in
    *_crediticl_auto_venv*) : ;;   # already registered; sourcing twice is harmless
    *) PROMPT_COMMAND="_crediticl_auto_venv${PROMPT_COMMAND:+;$PROMPT_COMMAND}" ;;
esac

# --- installer --------------------------------------------------------------
# Only runs when this file is EXECUTED (bash shell_hook.sh), never when sourced.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -uo pipefail

    HOOK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    MARKER="# CreditICL venv hook"
    LINE="[ -f \"${HOOK}\" ] && source \"${HOOK}\"   ${MARKER}"
    BASHRC="${HOME}/.bashrc"

    case "${1:---install}" in
        --install)
            if grep -qF "${MARKER}" "${BASHRC}" 2>/dev/null; then
                echo "Already installed in ${BASHRC}:"
                grep -nF "${MARKER}" "${BASHRC}"
            else
                # Newline first, in case .bashrc does not end with one.
                printf '\n%s\n' "${LINE}" >> "${BASHRC}"
                echo "Added to ${BASHRC}:"
                echo "  ${LINE}"
            fi
            echo
            echo "Now run:   source ~/.bashrc"
            echo "Then:      cd \$VSC_DATA/CreditICL && python -c 'import sys; print(sys.prefix)'"
            echo "It should print a path ending in .venv-<arch>."
            ;;
        --uninstall)
            if grep -qF "${MARKER}" "${BASHRC}" 2>/dev/null; then
                # Keep a backup: editing someone's .bashrc in place is not something to do
                # without a way back.
                cp "${BASHRC}" "${BASHRC}.crediticl.bak"
                grep -vF "${MARKER}" "${BASHRC}.crediticl.bak" > "${BASHRC}"
                echo "Removed. Backup at ${BASHRC}.crediticl.bak"
            else
                echo "Not installed; nothing to remove."
            fi
            ;;
        --status)
            echo "hook file : ${HOOK}"
            echo "in bashrc : $(grep -cF "${MARKER}" "${BASHRC}" 2>/dev/null || echo 0) line(s)"
            echo "repo      : ${VSC_DATA:-\$VSC_DATA unset}/CreditICL"
            echo "arch      : ${VSC_ARCH_LOCAL:-unset}"
            echo "venvs found:"
            for v in "${VSC_DATA:-}/CreditICL"/.venv-* "${VSC_DATA:-}/CreditICL/.venv"; do
                [[ -x "$v/bin/python" ]] && echo "  $v  (module: $(cat "$v/.python_module" 2>/dev/null || echo '?'))"
            done
            echo "active now: ${VIRTUAL_ENV:-none}"
            ;;
        *)
            echo "usage: bash $0 [--install|--uninstall|--status]" >&2
            exit 2
            ;;
    esac
fi
