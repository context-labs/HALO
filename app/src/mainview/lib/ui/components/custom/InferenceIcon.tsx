const CLIP_PATH_ID = "inference-icon";

type InferenceIconProps = {
  height?: number | string;
  width?: number | string;
};

export function InferenceIcon({ height, width }: InferenceIconProps) {
  return (
    <svg
      fill="none"
      height={height ?? 116}
      viewBox="0 0 880 116"
      width={width ?? 880}
      xmlns="http://www.w3.org/2000/svg"
    >
      <g clipPath={`url(#${CLIP_PATH_ID})`}>
        <path
          d="M209.001 67.2804L161.875 2.466H135.667V113.515H163.341V47.6802L212.101 113.515H236.694V2.466H209.001V67.2804Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M253.788 113.515H282.057V69.9346H320.348V46.2139H282.057V26.1867H321.832V2.466H253.788V113.515Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M334.788 113.515H404.15V89.8133H363.075V68.5983H402.684V44.8961H363.075V26.1867H404.15V2.466H334.788V113.515Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M499.738 38.1214C499.738 15.44 481.623 2.48456 455.266 2.48456H418.59V113.534H447.155V73.7582L477.502 113.534H512.267L476.927 69.4892C490.773 63.8838 499.757 52.9886 499.757 38.1214H499.738ZM455.118 52.2462H447.174V25.8898H455.118C464.695 25.8898 470.431 31.0497 470.431 38.9937C470.431 46.9378 464.695 52.2462 455.118 52.2462Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M516.239 113.515H585.619V89.8133H544.525V68.5983H584.134V44.8961H544.525V26.1867H585.619V2.466H516.239V113.515Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M673.393 67.2804L626.267 2.466H600.041V113.515H627.734V47.6802L676.474 113.515H701.086V2.466H673.393V67.2804Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M712.557 58C712.557 89.5163 738.635 115.297 771.023 115.297C780.007 115.297 788.99 113.534 796.656 109.989V78.0271C791.347 83.7624 782.512 88.1985 771.617 88.1985C754.096 88.1985 741.419 75.3915 741.419 58C741.419 40.6085 754.077 27.8015 771.617 27.8015C782.512 27.8015 791.347 32.219 796.656 37.9729V6.0111C788.99 2.63302 780.007 0.702698 771.023 0.702698C738.616 0.702698 712.557 26.4837 712.557 58Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M879.289 26.1867V2.466H809.927V113.515H879.289V89.8133H838.195V68.5983H877.822V44.8961H838.195V26.1867H879.289Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path
          d="M118.573 2.466H89.7107V113.515H118.573V2.466Z"
          className={`
            fill-black

            dark:fill-white
          `}
        />
        <path d="M75.029 2.466H54.5564V113.515H75.029V2.466Z" fill="#FF4405" />
        <path
          d="M40.8214 2.466H25.1189V113.515H40.8214V2.466Z"
          fill="#FAC515"
        />
        <path
          d="M11.9407 2.466H0.711426V113.515H11.9407V2.466Z"
          fill="#53B1FD"
        />
      </g>
      <defs>
        <clipPath id={CLIP_PATH_ID}>
          <rect
            className={`
              fill-white

              dark:fill-black
            `}
            height="114.595"
            transform="translate(0.711426 0.702698)"
            width="878.577"
          />
        </clipPath>
      </defs>
    </svg>
  );
}
