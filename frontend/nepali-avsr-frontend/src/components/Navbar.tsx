"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import "@/styles/components/_navbar.scss";

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false);
    const pathname = usePathname(); // get current route

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 0);
        };
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    return (
        <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
            <div className="nav-logo">
                <Link href="/">Shruti: Nepali AVSR</Link>
            </div>
            <ul className="nav-links">
                <li>
                    <Link
                        href="/realtime"
                        className={pathname === "/realtime" ? "active" : ""}
                    >
                        Realtime
                    </Link>
                </li>
                <li>
                    <Link
                        href="/offline"
                        className={pathname === "/offline" ? "active" : ""}
                    >
                        Offline
                    </Link>
                </li>
                <li>
                    <Link
                        href="/contact"
                        className={pathname === "/contact" ? "active" : ""}
                    >
                        Contact Us
                    </Link>
                </li>
            </ul>
        </nav>
    );
}
