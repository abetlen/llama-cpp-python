#!/usr/bin/env -S sh -c 'nvchecker -cnvchecker.toml --logger=json | jq -r '\''.version | sub("^v"; "") | split("-") | .[-1]'\'' | xargs -i{} sed -i "s/^\\(pkgver=\\).*/\\1{}/" $0'
# shellcheck shell=bash disable=SC2034,SC2154
# ex: nowrap
# Maintainer: Wu Zhenyu <wuzhenyu@ustc.edu>
pkgname=llama-cpp-python
pkgver=0.0.1
pkgrel=1
pkgdesc=""
arch=(x86 x86_64 arm aarch64)
url=https://github.com/Freed-Wu/$pkgname
license=(GPL3)
source=("$url/archive/$pkgver.tar.gz")
sha256sums=(SKIP)

package() {
	cd "$pkgname-$pkgver" || return 1

	make DESTDIR="$pkgdir" install
}
