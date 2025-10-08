"use client";

import Link from "next/link";
import "@/styles/components/_navbar.scss"

const Navbar=()=>{
    return(
        <nav className="navBar">
        <div className="nav-logo">
            Shruti: Nepali AVSR
        </div>
            <ul>
                <li><Link href={"/"}>Home</Link></li>
                <li><Link href={"/realtime"}>Realtime AVSR</Link></li>
                <li><Link href={"/offline"}>Offline AVSR</Link></li>
            </ul>
        </nav>
    );
}
export default Navbar;